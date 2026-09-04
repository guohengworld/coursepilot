import { ref } from 'vue'

/**
 * RAG 问答 SSE 流。
 *
 * 后端 /ask/stream 现在按 JSON 信封逐事件输出（每事件一行 data:，物理单行）：
 *   data: {"t":"tok","c":"<回答增量>"}
 *   data: {"t":"meta","citation_map":{...},"trace_id":...,...}   （流结束前，可选）
 *   data: {"t":"done"}
 * 兼容旧版裸文本 token 与 "[DONE]" 结束标记。
 */

interface StreamMetaPayload {
  t: 'meta'
  citation_map?: Record<string, unknown>
  trace_id?: string
  rewritten_query?: string
  source_kp_paths?: string[]
  top_scores?: number[]
  [k: string]: unknown
}

type StreamPayload =
  | { t: 'tok'; c: string }
  | StreamMetaPayload
  | { t: 'done' }

export function useSSE() {
  const isStreaming = ref(false)
  const abortController = ref<AbortController | null>(null)

  async function streamAsk(
    courseId: string,
    question: string,
    tokenGetter: () => string | null,
    onToken: (token: string) => void,
    onDone: () => void,
    onError: (err: string) => void,
    onMeta?: (payload: StreamMetaPayload) => void,
  ) {
    if (isStreaming.value) return
    isStreaming.value = true
    abortController.value = new AbortController()

    let doneFired = false
    const finish = () => {
      if (doneFired) return
      doneFired = true
      onDone()
      isStreaming.value = false
      abortController.value = null
    }

    function handlePayload(payload: string) {
      let parsed: StreamPayload | null = null
      if (payload.startsWith('{')) {
        try {
          parsed = JSON.parse(payload) as StreamPayload
        } catch {
          parsed = null
        }
      }
      if (parsed && typeof parsed.t === 'string') {
        if (parsed.t === 'tok' && typeof parsed.c === 'string') {
          onToken(parsed.c)
        } else if (parsed.t === 'meta') {
          onMeta?.(parsed as StreamMetaPayload)
        } else if (parsed.t === 'done') {
          finish()
        }
        return
      }
      // 兼容旧版裸文本 token / 结束标记
      if (payload === '[DONE]') {
        finish()
        return
      }
      onToken(payload)
    }

    try {
      const response = await fetch(`/api/v1/courses/${courseId}/ask/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokenGetter()}`,
        },
        body: JSON.stringify({ question }),
        signal: abortController.value.signal,
      })

      if (!response.ok) {
        onError(`HTTP ${response.status}: ${response.statusText}`)
        isStreaming.value = false
        return
      }

      const reader = response.body?.getReader()
      if (!reader) {
        onError('Response body is not readable')
        isStreaming.value = false
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (doneFired) break
          if (line.startsWith('data: ')) {
            handlePayload(line.slice(6))
          }
        }
        if (doneFired) break
      }

      if (!doneFired) finish()
    } catch (err: unknown) {
      if ((err as Error).name === 'AbortError') {
        finish()
      } else {
        onError((err as Error).message || 'Stream error')
      }
    } finally {
      isStreaming.value = false
    }
  }

  function stop() {
    if (abortController.value) {
      abortController.value.abort()
      abortController.value = null
    }
  }

  return { isStreaming, streamAsk, stop }
}
