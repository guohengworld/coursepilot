import { request } from './index'
import type {
  LoginRequest,
  RegisterRequest,
  LoginResponse,
  RegisterResponse,
  UserInfo,
} from '@/types'

export function login(data: LoginRequest) {
  return request<LoginResponse>('POST', '/auth/login', data)
}

export function register(data: RegisterRequest) {
  return request<RegisterResponse>('POST', '/auth/register', data)
}

export function getMe() {
  return request<UserInfo>('GET', '/auth/me')
}
