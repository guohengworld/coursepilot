import { request } from './index'
import type { QuizResponse, SubmitRequest, SubmitResponse } from '@/types'

export function getQuiz(sessionId: string) {
  return request<QuizResponse>('GET', `/practice/${sessionId}/quiz`)
}

export function submitAnswers(sessionId: string, data: SubmitRequest) {
  return request<SubmitResponse>('POST', `/practice/${sessionId}/submit`, data)
}
