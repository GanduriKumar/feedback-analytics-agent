import { useMemo, useState } from 'react';
import { Brain, KeyRound, Server } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import type { LLMProvider } from '../../types';

const providers: Array<{ value: LLMProvider; label: string; requiresKey: boolean; models: string[] }> = [
  { value: 'ollama', label: 'Ollama (local)', requiresKey: false, models: ['mistral', 'llama3', 'qwen2.5'] },
  { value: 'openai', label: 'OpenAI', requiresKey: true, models: ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini'] },
  { value: 'anthropic', label: 'Anthropic', requiresKey: true, models: ['claude-3-5-sonnet', 'claude-3-opus'] },
  { value: 'gemini', label: 'Gemini', requiresKey: true, models: ['gemini-1.5-pro', 'gemini-1.5-flash'] },
];

export function LLMConfig() {
  const { llmConfig, setLLMConfig } = useAppStore();
  const [showKey, setShowKey] = useState(false);

  const selected = useMemo(() => providers.find((p) => p.value === llmConfig.provider)!, [llmConfig.provider]);

  return (
    <section className="rounded-xl border border-google-gray-200 bg-white p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Brain className="w-5 h-5 text-google-blue-600" />
        <h3 className="text-base font-semibold text-google-gray-900">LLM Settings</h3>
        <span className="text-xs text-google-gray-600">(optional for now)</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-google-gray-700 mb-1">Provider</label>
          <select
            value={llmConfig.provider}
            onChange={(e) => {
              const provider = e.target.value as LLMProvider;
              const model = providers.find((p) => p.value === provider)?.models[0] || '';
              setLLMConfig({ ...llmConfig, provider, model });
            }}
            className="w-full rounded-lg border border-google-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-google-blue-500"
          >
            {providers.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-google-gray-700 mb-1">Model</label>
          <select
            value={llmConfig.model}
            onChange={(e) => setLLMConfig({ ...llmConfig, model: e.target.value })}
            className="w-full rounded-lg border border-google-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-google-blue-500"
          >
            {selected.models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        {selected.requiresKey && (
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-google-gray-700 mb-1 flex items-center gap-2">
              <KeyRound className="w-4 h-4" /> API Key
            </label>
            <div className="flex gap-2">
              <input
                type={showKey ? 'text' : 'password'}
                value={llmConfig.apiKey || ''}
                onChange={(e) => setLLMConfig({ ...llmConfig, apiKey: e.target.value })}
                placeholder="Paste your API key"
                className="flex-1 rounded-lg border border-google-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-google-blue-500"
              />
              <button
                type="button"
                onClick={() => setShowKey((v) => !v)}
                className="px-3 py-2 rounded-lg border border-google-gray-300 text-google-gray-700 hover:bg-google-gray-50"
              >
                {showKey ? 'Hide' : 'Show'}
              </button>
            </div>
            <p className="text-xs text-google-gray-600 mt-1">Stored locally in your browser (no IAM/RBAC).</p>
          </div>
        )}

        {llmConfig.provider === 'ollama' && (
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-google-gray-700 mb-1 flex items-center gap-2">
              <Server className="w-4 h-4" /> Ollama Base URL
            </label>
            <input
              value={llmConfig.baseUrl || ''}
              onChange={(e) => setLLMConfig({ ...llmConfig, baseUrl: e.target.value })}
              placeholder="http://localhost:11434"
              className="w-full rounded-lg border border-google-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-google-blue-500"
            />
          </div>
        )}
      </div>
    </section>
  );
}
