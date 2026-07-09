<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { getCourses } from '@/api/courses'

const auth = useAuthStore()
const router = useRouter()
const courseCount = ref(0)
const loading = ref(true)

onMounted(async () => {
  const res = await getCourses()
  if (res.ok && Array.isArray(res.data)) {
    courseCount.value = (res.data as any[]).length
  }
  loading.value = false
})

const quickActions = [
  { title: '课程管理', desc: '创建和管理课程', path: '/courses', icon: 'Notebook', color: '#409eff' },
  { title: '知识点树', desc: '浏览知识点结构', path: '/knowledge-points', icon: 'Share', color: '#67c23a' },
  { title: 'RAG 问答', desc: '基于教材内容提问', path: '/rag-qa', icon: 'ChatLineSquare', color: '#e6a23c' },
  { title: 'Agent 对话', desc: 'AI 智能教学助手', path: '/agent', icon: 'MagicStick', color: '#9b59b6' },
  { title: 'API 控制台', desc: '测试所有 API 端点', path: '/api-console', icon: 'Tools', color: '#f56c6c' },
]
</script>

<template>
  <div class="dashboard">
    <div class="welcome-section">
      <h1>欢迎回来，{{ auth.user?.username }}</h1>
      <p class="text-secondary">角色：<el-tag :type="auth.isSuperuser ? 'danger' : 'info'" size="small">{{ auth.user?.role }}</el-tag></p>
    </div>

    <el-row :gutter="20" class="stats-row">
      <el-col :span="8">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-value">{{ loading ? '...' : courseCount }}</div>
          <div class="stat-label">课程数量</div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-value">{{ auth.isLoggedIn ? '✓' : '✗' }}</div>
          <div class="stat-label">登录状态</div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-value">{{ auth.isSuperuser ? '超级管理员' : auth.isTeacher ? '教师' : '学生' }}</div>
          <div class="stat-label">用户角色</div>
        </el-card>
      </el-col>
    </el-row>

    <h3 class="section-title">快捷操作</h3>
    <el-row :gutter="20">
      <el-col :span="6" v-for="item in quickActions" :key="item.path">
        <el-card
          shadow="hover"
          class="action-card"
          @click="router.push(item.path)"
          style="cursor: pointer"
        >
          <el-icon :size="28" :color="item.color">
            <component :is="item.icon" />
          </el-icon>
          <div class="action-title">{{ item.title }}</div>
          <div class="action-desc">{{ item.desc }}</div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.dashboard {
  max-width: 1200px;
}

.welcome-section {
  margin-bottom: 24px;
}

.welcome-section h1 {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 4px;
}

.stats-row {
  margin-bottom: 32px;
}

.stat-card {
  text-align: center;
  padding: 8px;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: var(--el-color-primary);
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: var(--el-text-color-secondary);
}

.section-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
}

.action-card {
  text-align: center;
  padding: 20px 8px;
  transition: transform 0.2s;
}

.action-card:hover {
  transform: translateY(-4px);
}

.action-title {
  margin-top: 12px;
  font-size: 15px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.action-desc {
  margin-top: 4px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
