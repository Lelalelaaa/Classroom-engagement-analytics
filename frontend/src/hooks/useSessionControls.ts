'use client';
import { useState } from 'react';
import { api } from '@/lib/api';
import { useEngagementStore } from '@/store/engagementStore';

export function useSessionControls() {
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const { setSessionId, clearSession } = useEngagementStore();

  const startSession = async (classroomId: string, teacherName?: string, subject?: string) => {
    setLoading(true);
    setError(null);
    try {
      const session = await api.createSession({ classroom_id: classroomId, teacher_name: teacherName, subject }) as { id: string };
      setSessionId(session.id);
      return session.id;
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to start session');
      return null;
    } finally {
      setLoading(false);
    }
  };

  const endSession = async (sessionId: string) => {
    setLoading(true);
    try {
      await api.endSession(sessionId);
      clearSession();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to end session');
    } finally {
      setLoading(false);
    }
  };

  return { startSession, endSession, loading, error };
}
