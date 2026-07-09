<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { register } from '@/api/auth'
import type { RegisterRequest } from '@/types'

const router = useRouter()

const form = reactive<RegisterRequest & { confirmPassword: string }>({
  username: '',
  password: '',
  confirmPassword: '',
})
const loading = ref(false)
const errorMsg = ref('')

async function handleRegister() {
  if (!form.username || !form.password) {
    errorMsg.value = '请填写所有字段'
    return
  }
  if (form.username.length < 3) {
    errorMsg.value = '用户名至少 3 个字符'
    return
  }
  if (form.password.length < 8) {
    errorMsg.value = '密码至少 8 个字符'
    return
  }
  if (form.password !== form.confirmPassword) {
    errorMsg.value = '两次输入的密码不一致'
    return
  }

  loading.value = true
  errorMsg.value = ''

  const res = await register({ username: form.username, password: form.password })
  if (res.ok) {
    ElMessage.success('注册成功！请登录')
    router.push('/login')
  } else {
    errorMsg.value = (res.data as any)?.detail || '注册失败'
  }
  loading.value = false
}
</script>

<template>
  <div class="auth-page">
    <el-card class="auth-card" shadow="always">
      <div class="auth-header">
        <el-icon :size="32" color="var(--el-color-primary)"><School /></el-icon>
        <h2>创建账号</h2>
        <p class="text-secondary">注册 CoursePilot 账号</p>
      </div>

      <el-form
        @submit.prevent="handleRegister"
        :model="form"
        label-position="top"
        class="auth-form"
      >
        <el-form-item label="用户名">
          <el-input
            v-model="form.username"
            placeholder="3-64 个字符"
            :prefix-icon="'User'"
          />
        </el-form-item>

        <el-form-item label="密码">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="至少 8 个字符"
            show-password
            :prefix-icon="'Lock'"
          />
        </el-form-item>

        <el-form-item label="确认密码">
          <el-input
            v-model="form.confirmPassword"
            type="password"
            placeholder="再次输入密码"
            show-password
            :prefix-icon="'Lock'"
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
          {{ loading ? '注册中...' : '注 册' }}
        </el-button>
      </el-form>

      <div class="auth-footer">
        已有账号？
        <router-link to="/login">去登录</router-link>
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
