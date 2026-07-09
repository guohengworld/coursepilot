import { request } from './index'
import type { ChatRequest, ChatResponse, SessionStatus } from '@/types'

export function chat(data: ChatRequest) {
  return request<ChatResponse>('POST', '/agent/chat', data)
}

export function getSession(sessionId: string) {
  return request<SessionStatus>('GET', `/agent/sessions/${sessionId}`)
}

export function approveSession(sessionId: string) {
  return request<{ status: string; session_id: string; answer: string }>(
    'POST',
    `/agent/sessions/${sessionId}/approve`,
  )
}
