import { request } from './index'
import type { ChatAcceptedResponse, SessionPollResponse, SessionListItem } from '@/types'

export function chat(data: { message: string; course_id: string; session_id?: string }) {
  return request<ChatAcceptedResponse>('POST', '/agent/chat', data)
}

export function getSession(sessionId: string) {
  return request<SessionPollResponse>('GET', `/agent/sessions/${sessionId}`)
}

export function listSessions() {
  return request<SessionListItem[]>('GET', '/agent/sessions')
}

export function deleteSession(sessionId: string) {
  return request<null>('DELETE', `/agent/sessions/${sessionId}`)
}

export function approveSession(sessionId: string) {
  return request<{ status: string; session_id: string; answer: string }>(
    'POST',
    `/agent/sessions/${sessionId}/approve`,
  )
}
