import { ref } from 'vue'

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
  ) {
    if (isStreaming.value) return
    isStreaming.value = true
    abortController.value = new AbortController()

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
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') {
              onDone()
              isStreaming.value = false
              return
            }
            onToken(data)
          }
        }
      }

      onDone()
    } catch (err: unknown) {
      if ((err as Error).name === 'AbortError') {
        onDone()
      } else {
        onError((err as Error).message || 'Stream error')
      }
    }

    isStreaming.value = false
  }

  function stop() {
    if (abortController.value) {
      abortController.value.abort()
      abortController.value = null
    }
  }

  return { isStreaming, streamAsk, stop }
}
