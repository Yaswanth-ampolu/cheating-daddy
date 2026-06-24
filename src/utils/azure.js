const OpenAI = require('openai');
const { getSystemPrompt } = require('./prompts');
const { sendToRenderer, initializeNewSession, saveConversationTurn, saveScreenAnalysis } = require('./gemini');

const TRANSCRIPTION_API_VERSION = '2025-04-01-preview';

const CHAT_DEPLOYMENTS = {
    'gpt-4.1-mini': 'gpt-4.1-mini',
    'gpt-5.1': 'gpt-5.1',
};

const VAD_MODES = {
    NORMAL: { energyThreshold: 0.01, speechFramesRequired: 3, silenceFramesRequired: 30 },
    LOW_BITRATE: { energyThreshold: 0.008, speechFramesRequired: 4, silenceFramesRequired: 35 },
    AGGRESSIVE: { energyThreshold: 0.015, speechFramesRequired: 2, silenceFramesRequired: 20 },
    VERY_AGGRESSIVE: { energyThreshold: 0.02, speechFramesRequired: 2, silenceFramesRequired: 15 },
};

let vadConfig = VAD_MODES.VERY_AGGRESSIVE;
let azureClient = null;
let azureBaseUrl = null;
let azureClassicBaseUrl = null;
let azureApiKey = null;
let azureModelChoice = 'gpt-4.1-mini';
let azureTranscriptionDeployment = 'gpt-4o-transcribe-diarize';
let azureLanguage = 'en';
let azureConversationHistory = [];
let currentSystemPrompt = null;
let isAzureActive = false;
let isSpeaking = false;
let speechBuffers = [];
let silenceFrameCount = 0;
let speechFrameCount = 0;
let resampleRemainder = Buffer.alloc(0);
let activeStream = null;
let turnQueue = Promise.resolve();

function normalizeAzureBaseUrl(resourceOrEndpoint) {
    const raw = (resourceOrEndpoint || '').trim().replace(/\/+$/, '');
    if (!raw) return '';

    if (/^https?:\/\//i.test(raw)) {
        if (/\/openai\/v1$/i.test(raw)) {
            return raw;
        }
        if (/\/openai$/i.test(raw)) {
            return `${raw}/v1`;
        }
        return `${raw}/openai/v1`;
    }

    const resourceName = raw
        .replace(/^https?:\/\//i, '')
        .split('/')[0]
        .split('.')[0];
    return `https://${resourceName}.services.ai.azure.com/openai/v1`;
}

function normalizeAzureClassicBaseUrl(resourceOrEndpoint) {
    const raw = (resourceOrEndpoint || '').trim().replace(/\/+$/, '');
    if (!raw) return '';

    let host = raw;
    if (/^https?:\/\//i.test(raw)) {
        host = new URL(raw).hostname;
    } else {
        host = raw.split('/')[0];
    }

    if (host.endsWith('.openai.azure.com')) {
        return `https://${host}`;
    }

    const resourceName = host.split('.')[0];
    return `https://${resourceName}.openai.azure.com`;
}

function normalizeLanguage(language) {
    if (!language || typeof language !== 'string') return 'en';
    return language.split('-')[0].toLowerCase();
}

function getAnswerDeployment(modelChoice = azureModelChoice) {
    return CHAT_DEPLOYMENTS[modelChoice] || CHAT_DEPLOYMENTS['gpt-4.1-mini'];
}

function trimConversationHistory(maxTurns = 20) {
    if (azureConversationHistory.length > maxTurns) {
        azureConversationHistory = azureConversationHistory.slice(-maxTurns);
    }
}

function queueTurn(work) {
    const nextTurn = turnQueue.then(work, work);
    turnQueue = nextTurn.catch(() => {});
    return nextTurn;
}

function calculateRMS(pcm16Buffer) {
    const samples = pcm16Buffer.length / 2;
    if (samples === 0) return 0;

    let sumSquares = 0;
    for (let i = 0; i < samples; i++) {
        const sample = pcm16Buffer.readInt16LE(i * 2) / 32768;
        sumSquares += sample * sample;
    }

    return Math.sqrt(sumSquares / samples);
}

function resample24kTo16k(inputBuffer) {
    const combined = Buffer.concat([resampleRemainder, inputBuffer]);
    const inputSamples = Math.floor(combined.length / 2);
    const outputSamples = Math.floor((inputSamples * 2) / 3);
    const outputBuffer = Buffer.alloc(outputSamples * 2);

    for (let i = 0; i < outputSamples; i++) {
        const srcPos = (i * 3) / 2;
        const srcIndex = Math.floor(srcPos);
        const frac = srcPos - srcIndex;

        const s0 = combined.readInt16LE(srcIndex * 2);
        const s1 = srcIndex + 1 < inputSamples ? combined.readInt16LE((srcIndex + 1) * 2) : s0;
        const interpolated = Math.round(s0 + frac * (s1 - s0));
        outputBuffer.writeInt16LE(Math.max(-32768, Math.min(32767, interpolated)), i * 2);
    }

    const consumedInputSamples = Math.ceil((outputSamples * 3) / 2);
    const remainderStart = consumedInputSamples * 2;
    resampleRemainder = remainderStart < combined.length ? combined.slice(remainderStart) : Buffer.alloc(0);

    return outputBuffer;
}

function pcm16ToWav(pcm16Buffer, sampleRate = 16000, channels = 1) {
    const bitsPerSample = 16;
    const blockAlign = channels * (bitsPerSample / 8);
    const byteRate = sampleRate * blockAlign;
    const dataSize = pcm16Buffer.length;
    const wavBuffer = Buffer.alloc(44 + dataSize);

    wavBuffer.write('RIFF', 0);
    wavBuffer.writeUInt32LE(36 + dataSize, 4);
    wavBuffer.write('WAVE', 8);
    wavBuffer.write('fmt ', 12);
    wavBuffer.writeUInt32LE(16, 16);
    wavBuffer.writeUInt16LE(1, 20);
    wavBuffer.writeUInt16LE(channels, 22);
    wavBuffer.writeUInt32LE(sampleRate, 24);
    wavBuffer.writeUInt32LE(byteRate, 28);
    wavBuffer.writeUInt16LE(blockAlign, 32);
    wavBuffer.writeUInt16LE(bitsPerSample, 34);
    wavBuffer.write('data', 36);
    wavBuffer.writeUInt32LE(dataSize, 40);
    pcm16Buffer.copy(wavBuffer, 44);

    return wavBuffer;
}

function formatDiarizedTranscript(transcription) {
    if (!transcription) return '';
    if (!Array.isArray(transcription.segments) || transcription.segments.length === 0) {
        return (transcription.text || '').trim();
    }

    const speakerMap = new Map();

    return transcription.segments
        .map(segment => {
            const text = (segment.text || '').trim();
            if (!text) return '';

            const rawSpeaker = segment.speaker || 'A';
            if (!speakerMap.has(rawSpeaker)) {
                const speakerIndex = speakerMap.size;
                const label = String.fromCharCode(65 + Math.min(speakerIndex, 25));
                speakerMap.set(rawSpeaker, `Speaker ${label}`);
            }

            return `[${speakerMap.get(rawSpeaker)}]: ${text}`;
        })
        .filter(Boolean)
        .join('\n');
}

async function transcribeAudio(audioData) {
    if (!azureClient || !azureClassicBaseUrl) {
        return '';
    }

    const wavBuffer = pcm16ToWav(audioData, 16000, 1);
    const form = new FormData();
    form.append('file', new Blob([wavBuffer], { type: 'audio/wav' }), 'segment.wav');
    form.append('response_format', 'diarized_json');
    if (azureLanguage) {
        form.append('language', azureLanguage);
    }

    const endpoint = `${azureClassicBaseUrl}/openai/deployments/${encodeURIComponent(azureTranscriptionDeployment || 'gpt-4o-transcribe-diarize')}/audio/transcriptions?api-version=${TRANSCRIPTION_API_VERSION}`;
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
            'api-key': azureApiKey,
        },
        body: form,
    });

    if (!response.ok) {
        throw new Error(`Azure transcription error ${response.status}: ${await response.text()}`);
    }

    const result = await response.json();
    return formatDiarizedTranscript(result);
}

function buildHistoryInput(userText) {
    const history = azureConversationHistory.map(message => ({
        role: message.role,
        content: message.content,
    }));

    history.push({
        role: 'user',
        content: userText,
    });

    return history;
}

async function streamAzureResponse({ userText, input, saveTurn = true }) {
    if (!azureClient || !isAzureActive) {
        return { success: false, error: 'No active Azure session' };
    }

    const deployment = getAnswerDeployment();
    let fullText = '';
    let isFirst = true;

    try {
        activeStream = azureClient.responses.stream({
            model: deployment,
            instructions: currentSystemPrompt || 'You are a helpful assistant.',
            input,
        });

        activeStream.on('response.output_text.delta', event => {
            if (!event?.delta) return;
            fullText += event.delta;
            sendToRenderer(isFirst ? 'new-response' : 'update-response', fullText);
            isFirst = false;
        });

        const finalResponse = await activeStream.finalResponse();
        if (!fullText && typeof finalResponse?.output_text === 'string') {
            fullText = finalResponse.output_text;
            sendToRenderer('new-response', fullText);
        }

        const cleanedText = fullText.trim();

        if (saveTurn && cleanedText) {
            azureConversationHistory.push({ role: 'user', content: userText });
            azureConversationHistory.push({ role: 'assistant', content: cleanedText });
            trimConversationHistory();
            saveConversationTurn(userText, cleanedText);
        }

        sendToRenderer('update-status', 'Listening...');
        return { success: true, text: cleanedText, model: deployment };
    } catch (error) {
        console.error('[Azure] Response error:', error);
        sendToRenderer('update-status', 'Azure error: ' + error.message);
        return { success: false, error: error.message };
    } finally {
        activeStream = null;
    }
}

async function handleSpeechEnd(audioData) {
    if (!isAzureActive) return;

    if (audioData.length < 16000) {
        sendToRenderer('update-status', 'Listening...');
        return;
    }

    try {
        sendToRenderer('update-status', 'Transcribing...');
        const transcription = await transcribeAudio(audioData);

        if (!transcription || transcription.trim().length < 2) {
            sendToRenderer('update-status', 'Listening...');
            return;
        }

        sendToRenderer('update-status', 'Generating response...');
        await streamAzureResponse({
            userText: transcription.trim(),
            input: buildHistoryInput(transcription.trim()),
            saveTurn: true,
        });
    } catch (error) {
        console.error('[Azure] Speech handling error:', error);
        sendToRenderer('update-status', 'Azure transcription error: ' + error.message);
    }
}

function processVAD(pcm16kBuffer) {
    const rms = calculateRMS(pcm16kBuffer);
    const isVoice = rms > vadConfig.energyThreshold;

    if (isVoice) {
        speechFrameCount++;
        silenceFrameCount = 0;

        if (!isSpeaking && speechFrameCount >= vadConfig.speechFramesRequired) {
            isSpeaking = true;
            speechBuffers = [];
            sendToRenderer('update-status', 'Listening... (speech detected)');
        }
    } else {
        silenceFrameCount++;
        speechFrameCount = 0;

        if (isSpeaking && silenceFrameCount >= vadConfig.silenceFramesRequired) {
            isSpeaking = false;
            const audioData = Buffer.concat(speechBuffers);
            speechBuffers = [];
            queueTurn(() => handleSpeechEnd(audioData));
            return;
        }
    }

    if (isSpeaking) {
        speechBuffers.push(Buffer.from(pcm16kBuffer));
    }
}

async function initializeAzureSession({
    apiKey,
    resourceOrEndpoint,
    modelChoice = 'gpt-4.1-mini',
    transcriptionDeployment = 'gpt-4o-transcribe-diarize',
    customPrompt = '',
    profile = 'interview',
    language = 'en-US',
}) {
    sendToRenderer('session-initializing', true);

    try {
        azureBaseUrl = normalizeAzureBaseUrl(resourceOrEndpoint);
        azureClassicBaseUrl = normalizeAzureClassicBaseUrl(resourceOrEndpoint);
        azureApiKey = apiKey;
        azureModelChoice = modelChoice;
        azureTranscriptionDeployment = transcriptionDeployment;
        azureLanguage = normalizeLanguage(language);
        currentSystemPrompt = getSystemPrompt(profile, customPrompt, false);
        azureClient = new OpenAI({
            apiKey,
            baseURL: azureBaseUrl,
        });

        azureConversationHistory = [];
        isSpeaking = false;
        speechBuffers = [];
        silenceFrameCount = 0;
        speechFrameCount = 0;
        resampleRemainder = Buffer.alloc(0);
        turnQueue = Promise.resolve();
        initializeNewSession(profile, customPrompt);
        isAzureActive = true;

        sendToRenderer('session-initializing', false);
        sendToRenderer('update-status', 'Azure ready - Listening...');
        return true;
    } catch (error) {
        console.error('[Azure] Initialization error:', error);
        isAzureActive = false;
        azureClient = null;
        sendToRenderer('session-initializing', false);
        sendToRenderer('update-status', 'Azure init error: ' + error.message);
        return false;
    }
}

function processAzureAudio(monoChunk24k) {
    if (!isAzureActive) return;
    const pcm16k = resample24kTo16k(monoChunk24k);
    if (pcm16k.length > 0) {
        processVAD(pcm16k);
    }
}

async function sendAzureText(text) {
    if (!isAzureActive || !azureClient) {
        return { success: false, error: 'No active Azure session' };
    }

    const trimmedText = text.trim();
    if (!trimmedText) {
        return { success: false, error: 'Empty text message' };
    }

    sendToRenderer('update-status', 'Generating response...');
    return queueTurn(() =>
        streamAzureResponse({
            userText: trimmedText,
            input: buildHistoryInput(trimmedText),
            saveTurn: true,
        })
    );
}

async function sendAzureImage(base64Data, prompt) {
    if (!isAzureActive || !azureClient) {
        return { success: false, error: 'No active Azure session' };
    }

    const deployment = getAnswerDeployment();
    let fullText = '';
    let isFirst = true;

    try {
        sendToRenderer('update-status', 'Analyzing image...');

        activeStream = azureClient.responses.stream({
            model: deployment,
            instructions: currentSystemPrompt || 'You are a helpful assistant.',
            input: [
                {
                    role: 'user',
                    content: [
                        { type: 'input_text', text: prompt },
                        {
                            type: 'input_image',
                            image_url: `data:image/jpeg;base64,${base64Data}`,
                            detail: 'low',
                        },
                    ],
                },
            ],
        });

        activeStream.on('response.output_text.delta', event => {
            if (!event?.delta) return;
            fullText += event.delta;
            sendToRenderer(isFirst ? 'new-response' : 'update-response', fullText);
            isFirst = false;
        });

        const finalResponse = await activeStream.finalResponse();
        if (!fullText && typeof finalResponse?.output_text === 'string') {
            fullText = finalResponse.output_text;
            sendToRenderer('new-response', fullText);
        }

        if (fullText.trim()) {
            saveScreenAnalysis(prompt, fullText, deployment);
        }

        sendToRenderer('update-status', 'Listening...');
        return { success: true, text: fullText.trim(), model: deployment };
    } catch (error) {
        console.error('[Azure] Image error:', error);
        sendToRenderer('update-status', 'Azure image error: ' + error.message);
        return { success: false, error: error.message };
    } finally {
        activeStream = null;
    }
}

function closeAzureSession() {
    isAzureActive = false;
    isSpeaking = false;
    speechBuffers = [];
    silenceFrameCount = 0;
    speechFrameCount = 0;
    resampleRemainder = Buffer.alloc(0);
    azureConversationHistory = [];
    currentSystemPrompt = null;
    azureBaseUrl = null;
    azureClassicBaseUrl = null;
    azureApiKey = null;
    azureTranscriptionDeployment = 'gpt-4o-transcribe-diarize';
    turnQueue = Promise.resolve();

    if (activeStream && typeof activeStream.abort === 'function') {
        try {
            activeStream.abort();
        } catch (error) {
            console.warn('[Azure] Failed to abort active stream:', error.message);
        }
    }

    activeStream = null;
    azureClient = null;
}

module.exports = {
    initializeAzureSession,
    processAzureAudio,
    closeAzureSession,
    sendAzureText,
    sendAzureImage,
};
