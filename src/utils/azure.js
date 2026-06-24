const OpenAI = require('openai');
const { AzureOpenAI } = require('openai');
const { OpenAIRealtimeWebSocket } = require('openai/beta/realtime/websocket');
const { getSystemPrompt } = require('./prompts');
const { sendToRenderer, initializeNewSession, saveConversationTurn, saveScreenAnalysis } = require('./gemini');

const TRANSCRIPTION_API_VERSION = '2025-04-01-preview';
const AZURE_REALTIME_API_VERSION = '2025-04-01-preview';
const AZURE_REALTIME_LISTENING_STATUS = 'Azure live - Listening...';
const AZURE_BATCH_LISTENING_STATUS = 'Azure - Listening...';
const AZURE_RESPONSE_STYLE_SUFFIX = `

Azure answer style
-----
- Provide the exact words to say next.
- Stay concise, but do not under-answer.
- Prefer 2-4 short bullets or 2-4 short sentences when that gives a stronger answer.
- Lead with the strongest specific point first.
- Avoid coaching meta-commentary.
-----`;

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
let azureRealtimeClient = null;
let azureRealtimeSocket = null;
let azureBaseUrl = null;
let azureClassicBaseUrl = null;
let azureApiKey = null;
let azureModelChoice = 'gpt-4.1-mini';
let azureTranscriptionDeployment = 'gpt-4o-transcribe-diarize';
let azureRealtimeDeployment = '';
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
let azureRealtimeReady = false;
let azureRealtimeClosing = false;
let azureRealtimePartialTranscript = '';
let azureRealtimeProcessedItems = new Set();

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

function buildAzureSystemPrompt(profile, customPrompt) {
    return `${getSystemPrompt(profile, customPrompt, false)}${AZURE_RESPONSE_STYLE_SUFFIX}`;
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

function isAbortError(error) {
    const message = (error?.message || '').toLowerCase();
    return error?.name === 'AbortError' || message.includes('aborted') || message.includes('abort');
}

function resetBatchAudioState() {
    isSpeaking = false;
    speechBuffers = [];
    silenceFrameCount = 0;
    speechFrameCount = 0;
    resampleRemainder = Buffer.alloc(0);
}

function resetRealtimeState() {
    azureRealtimeReady = false;
    azureRealtimeClosing = false;
    azureRealtimePartialTranscript = '';
    azureRealtimeProcessedItems = new Set();
}

function setListeningStatus() {
    sendToRenderer('update-status', azureRealtimeReady ? AZURE_REALTIME_LISTENING_STATUS : AZURE_BATCH_LISTENING_STATUS);
}

function abortActiveResponse() {
    if (!activeStream || typeof activeStream.abort !== 'function') {
        return;
    }

    try {
        activeStream.abort();
    } catch (error) {
        console.warn('[Azure] Failed to abort active response:', error.message);
    }
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

function closeRealtimeSocket() {
    azureRealtimeClosing = true;

    if (azureRealtimeSocket) {
        try {
            azureRealtimeSocket.close({ code: 1000, reason: 'Session closing' });
        } catch (error) {
            console.warn('[Azure Realtime] Close warning:', error.message);
        }
    }

    azureRealtimeSocket = null;
    azureRealtimeClient = null;
    resetRealtimeState();
}

function fallbackToBatchMode(message) {
    if (!isAzureActive) return;

    console.warn('[Azure Realtime] Falling back to batch mode:', message);
    closeRealtimeSocket();
    resetBatchAudioState();
    sendToRenderer('update-status', `Azure live unavailable, using standard mode. ${message}`);
}

function buildRealtimeSessionUpdate() {
    return {
        type: 'session.update',
        session: {
            instructions: 'Transcribe the conversation accurately. Do not answer or speak.',
            modalities: ['text'],
            input_audio_format: 'pcm16',
            input_audio_transcription: {
                model: azureTranscriptionDeployment || 'gpt-4o-transcribe-diarize',
                ...(azureLanguage ? { language: azureLanguage } : {}),
            },
            turn_detection: {
                type: 'server_vad',
                create_response: false,
                interrupt_response: true,
                prefix_padding_ms: 300,
                silence_duration_ms: 450,
                threshold: 0.45,
            },
        },
    };
}

function getRealtimeFailureMessage(event) {
    if (event?.error?.message) {
        return event.error.message;
    }
    return 'Realtime transcription failed';
}

function registerRealtimeEventHandlers(realtimeSocket, complete) {
    realtimeSocket.on('error', error => {
        console.error('[Azure Realtime] Error:', error);
        if (!azureRealtimeReady) {
            closeRealtimeSocket();
            complete(false);
            return;
        }
        if (!azureRealtimeClosing) {
            fallbackToBatchMode(error.message || 'Realtime connection error');
        }
    });

    realtimeSocket.on('session.updated', () => {
        azureRealtimeReady = true;
        setListeningStatus();
        complete(true);
    });

    realtimeSocket.on('input_audio_buffer.speech_started', () => {
        abortActiveResponse();
        sendToRenderer('update-status', 'Azure live - Speech detected...');
    });

    realtimeSocket.on('input_audio_buffer.speech_stopped', () => {
        sendToRenderer('update-status', 'Azure live - Transcribing...');
    });

    realtimeSocket.on('conversation.item.input_audio_transcription.delta', event => {
        azureRealtimePartialTranscript += event.delta || '';
    });

    realtimeSocket.on('conversation.item.input_audio_transcription.failed', event => {
        fallbackToBatchMode(getRealtimeFailureMessage(event));
    });

    realtimeSocket.on('conversation.item.input_audio_transcription.completed', event => {
        const itemId = event.item_id || '';
        if (itemId && azureRealtimeProcessedItems.has(itemId)) {
            return;
        }
        if (itemId) {
            azureRealtimeProcessedItems.add(itemId);
        }

        const transcript = (event.transcript || azureRealtimePartialTranscript || '').trim();
        azureRealtimePartialTranscript = '';

        if (!transcript || transcript.length < 2) {
            setListeningStatus();
            return;
        }

        queueTurn(() =>
            streamAzureResponse({
                userText: transcript,
                input: buildHistoryInput(transcript),
                saveTurn: true,
            })
        );
    });
}

function bindRealtimeSocketLifecycle(realtimeSocket, complete) {
    const bindSocketEvent = (socket, eventName, handler) => {
        if (typeof socket.addEventListener === 'function') {
            socket.addEventListener(eventName, handler);
            return;
        }
        if (typeof socket.on === 'function') {
            socket.on(eventName, handler);
        }
    };

    bindSocketEvent(realtimeSocket.socket, 'open', () => {
        try {
            realtimeSocket.send(buildRealtimeSessionUpdate());
            setTimeout(() => {
                if (!azureRealtimeReady) {
                    azureRealtimeReady = true;
                    setListeningStatus();
                    complete(true);
                }
            }, 1500);
        } catch (error) {
            console.error('[Azure Realtime] Failed to configure session:', error);
            closeRealtimeSocket();
            complete(false);
        }
    });

    bindSocketEvent(realtimeSocket.socket, 'close', event => {
        if (!azureRealtimeReady) {
            closeRealtimeSocket();
            complete(false);
            return;
        }
        if (!azureRealtimeClosing) {
            fallbackToBatchMode(`Realtime socket closed (${event.code || 'unknown'})`);
        }
    });
}

async function initializeRealtimeSession() {
    if (!azureRealtimeDeployment || !azureRealtimeDeployment.trim()) {
        return false;
    }

    try {
        resetRealtimeState();
        azureRealtimeClient = new AzureOpenAI({
            endpoint: azureClassicBaseUrl,
            apiKey: azureApiKey,
            apiVersion: AZURE_REALTIME_API_VERSION,
        });

        const realtimeSocket = await OpenAIRealtimeWebSocket.azure(azureRealtimeClient, {
            deploymentName: azureRealtimeDeployment.trim(),
        });

        azureRealtimeSocket = realtimeSocket;
        azureRealtimeClosing = false;

        return await new Promise(resolve => {
            let settled = false;
            const timeout = setTimeout(() => {
                if (settled) return;
                settled = true;
                closeRealtimeSocket();
                resolve(false);
            }, 10000);

            const complete = success => {
                if (settled) return;
                settled = true;
                clearTimeout(timeout);
                resolve(success);
            };

            registerRealtimeEventHandlers(realtimeSocket, complete);
            bindRealtimeSocketLifecycle(realtimeSocket, complete);
        });
    } catch (error) {
        console.error('[Azure Realtime] Initialization failed:', error);
        closeRealtimeSocket();
        return false;
    }
}

async function streamAzureResponse({ userText, input, saveTurn = true }) {
    if (!azureClient || !isAzureActive) {
        return { success: false, error: 'No active Azure session' };
    }

    const deployment = getAnswerDeployment();
    let fullText = '';
    let isFirst = true;

    try {
        sendToRenderer('update-status', 'Generating response...');

        activeStream = azureClient.responses.stream({
            model: deployment,
            instructions: currentSystemPrompt || 'You are a helpful assistant.',
            input,
            max_output_tokens: 320,
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

        setListeningStatus();
        return { success: true, text: cleanedText, model: deployment };
    } catch (error) {
        if (isAbortError(error)) {
            setListeningStatus();
            return { success: false, aborted: true, error: error.message };
        }

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
        setListeningStatus();
        return;
    }

    try {
        sendToRenderer('update-status', 'Transcribing...');
        const transcription = await transcribeAudio(audioData);

        if (!transcription || transcription.trim().length < 2) {
            setListeningStatus();
            return;
        }

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

function processBatchVAD(pcm16kBuffer) {
    const rms = calculateRMS(pcm16kBuffer);
    const isVoice = rms > vadConfig.energyThreshold;

    if (isVoice) {
        speechFrameCount++;
        silenceFrameCount = 0;

        if (!isSpeaking && speechFrameCount >= vadConfig.speechFramesRequired) {
            isSpeaking = true;
            speechBuffers = [];
            abortActiveResponse();
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
    realtimeDeployment = '',
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
        azureRealtimeDeployment = (realtimeDeployment || '').trim();
        azureLanguage = normalizeLanguage(language);
        currentSystemPrompt = buildAzureSystemPrompt(profile, customPrompt);
        azureClient = new OpenAI({
            apiKey,
            baseURL: azureBaseUrl,
        });

        azureConversationHistory = [];
        resetBatchAudioState();
        resetRealtimeState();
        turnQueue = Promise.resolve();
        initializeNewSession(profile, customPrompt);
        isAzureActive = true;

        if (azureRealtimeDeployment) {
            const realtimeReady = await initializeRealtimeSession();
            if (!realtimeReady) {
                console.warn('[Azure] Realtime unavailable, using batch mode');
            }
        }

        sendToRenderer('session-initializing', false);
        setListeningStatus();
        return true;
    } catch (error) {
        console.error('[Azure] Initialization error:', error);
        isAzureActive = false;
        azureClient = null;
        closeRealtimeSocket();
        sendToRenderer('session-initializing', false);
        sendToRenderer('update-status', 'Azure init error: ' + error.message);
        return false;
    }
}

function processAzureAudio(monoChunk24k) {
    if (!isAzureActive) return;

    if (azureRealtimeReady && azureRealtimeSocket) {
        try {
            azureRealtimeSocket.send({
                type: 'input_audio_buffer.append',
                audio: monoChunk24k.toString('base64'),
            });
            return;
        } catch (error) {
            console.error('[Azure Realtime] Audio append failed:', error);
            fallbackToBatchMode(error.message || 'Unable to append audio');
        }
    }

    const pcm16k = resample24kTo16k(monoChunk24k);
    if (pcm16k.length > 0) {
        processBatchVAD(pcm16k);
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
            max_output_tokens: 320,
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

        setListeningStatus();
        return { success: true, text: fullText.trim(), model: deployment };
    } catch (error) {
        if (isAbortError(error)) {
            setListeningStatus();
            return { success: false, aborted: true, error: error.message };
        }

        console.error('[Azure] Image error:', error);
        sendToRenderer('update-status', 'Azure image error: ' + error.message);
        return { success: false, error: error.message };
    } finally {
        activeStream = null;
    }
}

function closeAzureSession() {
    isAzureActive = false;
    resetBatchAudioState();
    closeRealtimeSocket();
    azureConversationHistory = [];
    currentSystemPrompt = null;
    azureBaseUrl = null;
    azureClassicBaseUrl = null;
    azureApiKey = null;
    azureModelChoice = 'gpt-4.1-mini';
    azureTranscriptionDeployment = 'gpt-4o-transcribe-diarize';
    azureRealtimeDeployment = '';
    azureLanguage = 'en';
    turnQueue = Promise.resolve();

    abortActiveResponse();
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
