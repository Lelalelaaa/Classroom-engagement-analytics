const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

export const api = {
  // Sessions
  createSession: (body: { classroom_id: string; teacher_name?: string; subject?: string }) =>
    apiFetch('/api/sessions/', { method: 'POST', body: JSON.stringify(body) }),

  endSession: (sessionId: string) =>
    apiFetch(`/api/sessions/${sessionId}/end`, { method: 'PATCH' }),

  getSession: (sessionId: string) =>
    apiFetch(`/api/sessions/${sessionId}`),

  // Classrooms
  listClassrooms: () =>
    apiFetch<{ id: string; name: string; school_name: string; capacity: number }[]>('/api/classrooms/'),

  createClassroom: (body: { name: string; school_name: string; capacity: number }) =>
    apiFetch('/api/classrooms/', { method: 'POST', body: JSON.stringify(body) }),

  // Analytics
  getSessionMetrics: (sessionId: string, limit = 200) =>
    apiFetch(`/api/analytics/sessions/${sessionId}/metrics?limit=${limit}`),

  getClassroomsSummary: () =>
    apiFetch('/api/analytics/classrooms/summary'),
};
