<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import { getCourses, createCourse, deleteCourse } from '@/api/courses'
import type { Course } from '@/types'

const auth = useAuthStore()
const router = useRouter()
const courses = ref<Course[]>([])
const loading = ref(false)

// Create dialog
const dialogVisible = ref(false)
const form = ref({ name: '', description: '' })
const creating = ref(false)

async function loadCourses() {
  loading.value = true
  const res = await getCourses()
  if (res.ok && Array.isArray(res.data)) {
    courses.value = res.data as Course[]
  }
  loading.value = false
}

async function handleCreate() {
  if (!form.value.name) return
  creating.value = true
  const res = await createCourse({ name: form.value.name, description: form.value.description || undefined })
  if (res.ok) {
    ElMessage.success('课程创建成功')
    dialogVisible.value = false
    form.value = { name: '', description: '' }
    await loadCourses()
  } else {
    ElMessage.error((res.data as any)?.detail || '创建失败')
  }
  creating.value = false
}

async function handleDelete(id: string, name: string) {
  try {
    await ElMessageBox.confirm(`确定要删除课程「${name}」吗？此操作不可恢复。`, '确认删除', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning',
    })
    const res = await deleteCourse(id)
    if (res.ok) {
      ElMessage.success('课程已删除')
      await loadCourses()
    } else {
      ElMessage.error((res.data as any)?.detail || '删除失败')
    }
  } catch {
    // cancelled
  }
}

onMounted(loadCourses)
</script>

<template>
  <div class="page-container">
    <div class="page-header">
      <h2>课程管理</h2>
      <el-button v-if="auth.isTeacher" type="primary" @click="dialogVisible = true">
        <el-icon><Plus /></el-icon> 创建课程
      </el-button>
    </div>

    <el-table :data="courses" v-loading="loading" stripe style="width: 100%">
      <el-table-column prop="name" label="课程名称" min-width="180" />
      <el-table-column prop="description" label="描述" min-width="200" show-overflow-tooltip />
      <el-table-column label="创建时间" width="160">
        <template #default="{ row }">
          {{ new Date(row.created_at).toLocaleDateString('zh-CN') }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="200" fixed="right">
        <template #default="{ row }">
          <el-button size="small" @click="router.push(`/courses/${row.id}`)">详情</el-button>
          <el-button
            v-if="auth.isSuperuser"
            size="small"
            type="danger"
            @click="handleDelete(row.id, row.name)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-empty v-if="!loading && courses.length === 0" description="暂无课程" />

    <!-- Create Dialog -->
    <el-dialog v-model="dialogVisible" title="创建课程" width="500px">
      <el-form :model="form" label-position="top">
        <el-form-item label="课程名称" required>
          <el-input v-model="form.name" placeholder="请输入课程名称" />
        </el-form-item>
        <el-form-item label="描述（可选）">
          <el-input v-model="form.description" type="textarea" :rows="3" placeholder="请输入课程描述" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="creating" @click="handleCreate">创建</el-button>
      </template>
    </el-dialog>
  </div>
</template>

