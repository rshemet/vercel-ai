import {
    LanguageModelV1,
    LanguageModelV1CallOptions,
    LanguageModelV1FinishReason,
    LanguageModelV1StreamPart,
    LanguageModelV1CallWarning,
    LanguageModelV1Prompt,
    LanguageModelV1TextPart,
    LanguageModelV1ObjectGenerationMode,
  } from '@ai-sdk/provider';
  
  export interface CactusChatSettings {}
  
  import {
    CactusLM,
    CompletionParams,
    NativeCompletionResult,
    CactusOAICompatibleMessage,
  } from 'cactus-react-native';
  
  import RNFS from 'react-native-fs';
  
  
  const ModelCache = {
    _cache: new Map<string, { localPath: string }>(),
    getLocalPath(url: string): string {
      const filename =
        url.split('/').pop()?.replace(/[^a-zA-Z0-9.-]/g, '_') ?? 'model.gguf';
      return `${RNFS.DocumentDirectoryPath}/${filename}`;
    },
    async isDownloaded(url: string): Promise<boolean> {
      if (this._cache.has(url)) return true; // in-session cache
      const localPath = this.getLocalPath(url);
      const exists = await RNFS.exists(localPath); // check filesystem
      if (exists) {
        this.add(url, localPath);
        return true;
      }
      return false;
    },
    add(url: string, localPath: string) {
      this._cache.set(url, { localPath });
    },
    list(): Array<{ url: string; localPath: string }> {
      return Array.from(ModelCache._cache.entries()).map(
        ([url, data]: [string, { localPath: string }]) => ({
          url,
          ...data,
        }),
      );
    },
  };
  
  export enum ModelStatus {
    IDLE,
    DOWNLOADING,
    INITIALIZING,
    READY,
    ERROR,
  }
  
  export interface CactusChatLanguageModelConfig {
    provider: string;
    generateId?: () => string;
  }
  
  export class CactusChatLanguageModel implements LanguageModelV1 {
    readonly specificationVersion = 'v1';
    readonly provider: string;
    readonly modelId: string;
    readonly modelUrl: string;
    readonly defaultObjectGenerationMode: LanguageModelV1ObjectGenerationMode = undefined;
  
    private lm: CactusLM | null = null;
    private status: ModelStatus = ModelStatus.IDLE;
    private lastError: Error | null = null;
    private conversationHistory: CactusOAICompatibleMessage[] = [];
  
    constructor(
      modelUrl: string,
      private readonly settings: CactusChatSettings,
      private readonly config: CactusChatLanguageModelConfig,
    ) {
      this.modelUrl = modelUrl;
      // modelId is the model file path used in CactusLM.init().
      // We use modelId because it's a required parameter in LanguageModelV2.
      this.modelId = ModelCache.getLocalPath(this.modelUrl);
      this.provider = config.provider; // 'cactus'
    }
  
    public getStatus(): ModelStatus {
      return this.status;
    }
  
    public getLastError(): Error | null {
      return this.lastError; // useful for debugging when this.status is ModelStatus.ERROR
    }
  
    static async listDownloadedModels() {
      return ModelCache.list();
    }
  
    private async adaptMessagesToStatefulContext(
      fullPrompt: CactusOAICompatibleMessage[],
    ): Promise<CactusOAICompatibleMessage[]> {
      let divergent = fullPrompt.length < this.conversationHistory.length;
      if (!divergent) {
        for (let i = 0; i < this.conversationHistory.length; i++) {
          // Using JSON.stringify for a concise deep comparison.
          if (JSON.stringify(this.conversationHistory[i]) !== JSON.stringify(fullPrompt[i])) {
            divergent = true;
            break;
          }
        }
      }
  
      if (divergent) {
        await this.lm!.rewind();
        this.conversationHistory = [];
        return fullPrompt;
      } else {
        return fullPrompt.slice(this.conversationHistory.length);
      }
    }
  
    async downloadModel(
      options: { onProgress?: (p: number, bytesWritten?: number, contentLength?: number) => void } = {},
    ): Promise<string> {
      this.status = ModelStatus.DOWNLOADING;
      try {
        await RNFS.downloadFile({
          fromUrl: this.modelUrl,
          toFile: this.modelId,
          progress: (res: { bytesWritten: number; contentLength: number }) => {
            const progress = res.bytesWritten / res.contentLength;
            options.onProgress?.(progress, res.bytesWritten, res.contentLength);
          },
        }).promise;
  
        ModelCache.add(this.modelUrl, this.modelId);
        this.status = ModelStatus.IDLE;
        return this.modelId;
      } catch (e) {
        this.lastError = e as Error;
        this.status = ModelStatus.ERROR;
        throw e;
      }
    }
  
    async initialize(): Promise<void> {
      if (this.status === ModelStatus.READY) return;
      if (this.status === ModelStatus.INITIALIZING) return;
      const isDownloaded = await ModelCache.isDownloaded(this.modelUrl);
      if (!isDownloaded) {
        throw new Error('Model not downloaded. Call downloadModel() first.');
      }
  
      this.status = ModelStatus.INITIALIZING;
      try {
        const { lm, error } = await CactusLM.init({ model: this.modelId });
        if (error) throw error;
        this.lm = lm;
        this.status = ModelStatus.READY;
      } catch (e) {
        this.lastError = e as Error;
        this.status = ModelStatus.ERROR;
        throw e;
      }
    }
  
    private assertIsReady(): void {
      if (this.status !== ModelStatus.READY || !this.lm) {
        throw new Error(
          `Model not ready. Status: ${
            ModelStatus[this.status]
          }. Error: ${this.lastError?.message}`,
        );
      }
    }
  
    private convertToCactusMessages(
      prompt: LanguageModelV1Prompt,
    ): CactusOAICompatibleMessage[] {
      return prompt.map((message: any) => {
        if (message.role === 'system') {
          return { role: 'system', content: message.content };
        }
        const content = message.content
          .filter((part: any) => part.type === 'text')
          .map((part: any) => (part as LanguageModelV1TextPart).text)
          .join('');
        return { role: message.role, content };
      });
    }
  
    private getCactusParams(
      options: LanguageModelV1CallOptions,
    ): CompletionParams {
      const params: CompletionParams = {};
      if (options.temperature != null) params.temperature = options.temperature;
      if (options.maxTokens != null) params.n_predict = options.maxTokens;
      if (options.stopSequences != null) params.stop = options.stopSequences;
      return params;
    }
  
    get supportedUrls(): Record<string, RegExp[]> {return {}} // cactus is on-device only
  
    async doGenerate(
      options: LanguageModelV1CallOptions,
    ): Promise<{
      text?: string;
      reasoning?: string;
      files?: Array<{ data: string | Uint8Array; mimeType: string }>;
      toolCalls?: Array<any>;
      finishReason: LanguageModelV1FinishReason;
      usage: { promptTokens: number; completionTokens: number };
      rawCall: { rawPrompt: unknown; rawSettings: Record<string, unknown> };
      rawResponse?: { headers?: Record<string, string>; body?: unknown };
      request?: { body?: string };
      response?: { id?: string; timestamp?: Date; modelId?: string };
      warnings?: LanguageModelV1CallWarning[];
      providerMetadata?: Record<string, Record<string, any>>;
      sources?: Array<any>;
      logprobs?: any;
    }> {
      this.assertIsReady();
      const fullPrompt = this.convertToCactusMessages(options.prompt);
      const newMessages = await this.adaptMessagesToStatefulContext(fullPrompt);
  
      if (newMessages.length === 0) {
        return {
          text: '',
          finishReason: 'stop',
          usage: { promptTokens: 0, completionTokens: 0 },
          rawCall: { rawPrompt: options.prompt, rawSettings: {} },
          warnings: [],
        };
      }
  
      const params = this.getCactusParams(options);
      const result: NativeCompletionResult = await this.lm!.completion(
        newMessages,
        params,
      );
  
      this.conversationHistory.push(...newMessages, {
        role: 'assistant',
        content: result.content,
      });
  
      return {
        text: result.content,
        finishReason: 'stop' as const,
        usage: {
          promptTokens: result.tokens_evaluated,
          completionTokens: result.tokens_predicted,
        },
        rawCall: { rawPrompt: options.prompt, rawSettings: {} },
        warnings: [],
      };
    }
  
    async doStream(
      options: LanguageModelV1CallOptions,
    ): Promise<{
      stream: ReadableStream<LanguageModelV1StreamPart>;
      rawCall: { rawPrompt: unknown; rawSettings: Record<string, unknown> };
      rawResponse?: { headers?: Record<string, string> };
      request?: { body?: string };
      warnings?: LanguageModelV1CallWarning[];
    }> {
      this.assertIsReady();
      const fullPrompt = this.convertToCactusMessages(options.prompt);
      const newMessages = await this.adaptMessagesToStatefulContext(fullPrompt);
      const params = this.getCactusParams(options);
  
      const stream = new ReadableStream<LanguageModelV1StreamPart>({
        start: async controller => {
          if (newMessages.length === 0) {
            controller.enqueue({
              type: 'finish',
              finishReason: 'stop',
              usage: { promptTokens: 0, completionTokens: 0 },
            });
            controller.close();
            return;
          }
  
          let fullResponse = '';
          const completionPromise = this.lm!.completion(
            newMessages,
            params,
            data => {
              // This callback is invoked from the native side for each token
              fullResponse += data.token;
              controller.enqueue({ type: 'text-delta', textDelta: data.token });
            },
          );
  
          completionPromise
            .then(result => {
              controller.enqueue({
                type: 'finish',
                finishReason: 'stop',
                usage: {
                  promptTokens: result.tokens_evaluated,
                  completionTokens: result.tokens_predicted,
                },
              });
              this.conversationHistory.push(...newMessages, {
                role: 'assistant',
                content: fullResponse,
              });
              controller.close();
            })
            .catch(error => {
              controller.error(error);
            });
        },
      });
  
      return { 
        stream, 
        warnings: [],
        rawCall: { rawPrompt: options.prompt, rawSettings: {} }
      };
    }
  } 