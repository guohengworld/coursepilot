import { request } from './index'
import type { Course, CourseCreate, Document, KnowledgePoint, AskRequest, AskResponse } from '@/types'
import api from './index'

export function getCourses() {
  return request<Course[]>('GET', '/courses')
}

export function createCourse(data: CourseCreate) {
  return request<Course>('POST', '/courses', data)
}

export function getCourse(id: string) {
  return request<Course>('GET', `/courses/${id}`)
}

export function deleteCourse(id: string) {
  return request<{ deleted: string }>('DELETE', `/courses/${id}`)
}

export function enrollCourse(id: string) {
  return request<{ status: 'enrolled' | 'already_member'; role: string }>(
    'POST',
    `/courses/${id}/enroll`,
  )
}

export function getDocuments(courseId: string) {
  return request<Document[]>('GET', `/courses/${courseId}/documents`)
}

export function deleteDocument(courseId: string, docId: string) {
  return request<{ deleted: boolean }>('DELETE', `/courses/${courseId}/document/${docId}`)
}

export async function uploadDocument(
  courseId: string,
  file: File,
): Promise<{ ok: boolean; status: number; data: unknown }> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('course_id', courseId)
  try {
    const res = await api.post('/courses/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return { ok: true, status: res.status, data: res.data }
  } catch (error: unknown) {
    const axios = (await import('axios')).default
    if (axios.isAxiosError(error) && error.response) {
      return {
        ok: false,
        status: error.response.status,
        data: error.response.data,
      }
    }
    return { ok: false, status: 0, data: { detail: 'Upload failed' } }
  }
}

export function getKnowledgePoints(courseId: string, documentId?: string) {
  let path = `/courses/${courseId}/knowledge-points`
  if (documentId) {
    path += `?document_id=${documentId}`
  }
  return request<KnowledgePoint[]>('GET', path)
}

export function askQuestion(courseId: string, data: AskRequest) {
  return request<AskResponse>('POST', `/courses/${courseId}/ask`, data)
}
