<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { login } from '@/api/auth'
import type { LoginRequest } from '@/types'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()

const form = reactive<LoginRequest>({
  username: '',
  password: '',
})
const loading = ref(false)
const errorMsg = ref('')

async function handleLogin() {
  if (!form.username || !form.password) {
    errorMsg.value = '请输入用户名和密码'
    return
  }
  loading.value = true
  errorMsg.value = ''

  const res = await login(form)
  if (res.ok && !('detail' in res.data)) {
    const data = res.data as { token: string; expires_in: number; user: { id: string; username: string; role: string } }
    auth.setToken(data.token, data.expires_in)
    auth.setUser(data.user as any)

    const redirect = (route.query.redirect as string) || '/dashboard'
    router.push(redirect)
  } else {
    errorMsg.value = (res.data as any)?.detail || '登录失败，请检查用户名和密码'
  }
  loading.value = false
}
</script>

<template>
  <div class="auth-page">
    <el-card class="auth-card" shadow="always">
      <div class="auth-header">
        <el-icon :size="32" color="var(--el-color-primary)"><School /></el-icon>
        <h2>CoursePilot 登录</h2>
        <p class="text-secondary">AI 计算机科学课程教学助手</p>
      </div>

      <el-form
        @submit.prevent="handleLogin"
        :model="form"
        label-position="top"
        class="auth-form"
      >
        <el-form-item label="用户名">
          <el-input
            v-model="form.username"
            placeholder="请输入用户名"
            :prefix-icon="'User'"
            @keyup.enter="handleLogin"
          />
        </el-form-item>

        <el-form-item label="密码">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="请输入密码"
            show-password
            :prefix-icon="'Lock'"
            @keyup.enter="handleLogin"
          />
        </el-form-item>

        <el-alert
          v-if="errorMsg"
          :title="errorMsg"
          type="error"
          show-icon
          :closable="true"
          @close="errorMsg = ''"
          class="mb-16"
        />

        <el-button
          type="primary"
          native-type="submit"
          :loading="loading"
          class="auth-btn"
        >
          {{ loading ? '登录中...' : '登 录' }}
        </el-button>
      </el-form>

      <div class="auth-footer">
        还没有账号？
        <router-link to="/register">立即注册</router-link>
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.auth-page {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.auth-card {
  width: 400px;
}

.auth-header {
  text-align: center;
  margin-bottom: 32px;
}

.auth-header h2 {
  margin-top: 12px;
  font-size: 22px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.auth-header p {
  margin-top: 6px;
}

.auth-form {
  margin-bottom: 16px;
}

.auth-btn {
  width: 100%;
  margin-top: 8px;
}

.auth-footer {
  text-align: center;
  font-size: 14px;
  color: var(--el-text-color-secondary);
}

.auth-footer a {
  color: var(--el-color-primary);
  font-weight: 500;
}
</style>
