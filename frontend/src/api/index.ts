import axios from 'axios'
import type { AxiosInstance, AxiosResponse } from 'axios'
import type { ApiError } from '@/types'

const BASE = '/api/v1'

const api: AxiosInstance = axios.create({
  baseURL: BASE,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor: inject JWT
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor: handle 401
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('token_expires_at')
      // Only redirect if not already on login/register
      const hash = window.location.hash
      if (!hash.includes('login') && !hash.includes('register')) {
        window.location.hash = '#/login'
      }
    }
    return Promise.reject(error)
  },
)

export async function request<T>(
  method: 'GET' | 'POST' | 'PUT' | 'DELETE',
  path: string,
  data?: unknown,
  options?: { headers?: Record<string, string>; timeout?: number },
): Promise<{ ok: boolean; status: number; data: T | ApiError }> {
  try {
    const response: AxiosResponse<T> = await api.request({
      method,
      url: path,
      data: method === 'POST' || method === 'PUT' ? data : undefined,
      ...options,
    })
    return { ok: true, status: response.status, data: response.data }
  } catch (error: unknown) {
    if (axios.isAxiosError(error) && error.response) {
      return {
        ok: false,
        status: error.response.status,
        data: error.response.data as ApiError,
      }
    }
    return {
      ok: false,
      status: 0,
      data: { detail: 'Network error or request cancelled' },
    }
  }
}

export function upload(path: string, formData: FormData): Promise<{
  ok: boolean
  status: number
  data: unknown | ApiError
}> {
  return request('POST', path, undefined, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(() => {
    // Use raw axios for upload to let browser set Content-Type boundary
    return api
      .post(`${BASE}${path}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then((res) => ({ ok: true, status: res.status, data: res.data }))
      .catch((err) => {
        if (axios.isAxiosError(err) && err.response) {
          return {
            ok: false,
            status: err.response.status,
            data: err.response.data as ApiError,
          }
        }
        return { ok: false, status: 0, data: { detail: 'Upload failed' } }
      })
  })
}

export default api

// Re-export from submodules for convenience
export { login, register, getMe } from './auth'
export {
  getCourses,
  createCourse,
  getCourse,
  deleteCourse,
  getDocuments,
  deleteDocument,
  uploadDocument,
  getKnowledgePoints,
  askQuestion,
} from './courses'
export { chat, getSession, listSessions, approveSession, deleteSession } from './agent'
export {
  getTasks,
  getTaskCandidates,
  createTaskDraft,
  getTask,
  updateTask,
  publishTask,
} from './tasks'
