'use client';
import { useEffect, useRef, useCallback } from 'react';
import { useEngagementStore } from '@/store/engagementStore';
import { LiveMetrics, Alert } from '@/lib/types';

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000';

export function useEngagementSocket(sessionId: string | null) {
  const { setMetrics, addAlert, setConnected } = useEngagementStore();
  const wsRef     = useRef<WebSocket | null>(null);
  const retryRef  = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (!sessionId) return;

    const ws = new WebSocket(`${WS_BASE}/ws/dashboard/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onclose = () => {
      setConnected(false);
      // Auto-reconnect after 3 seconds
      retryRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => ws.close();

    ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'low_engagement') {
          addAlert(data as Alert);
        } else {
          setMetrics({ ...data, timestamp: Date.now() } as LiveMetrics);
        }
      } catch {
        // Ignore malformed messages
      }
    };
  }, [sessionId, setMetrics, addAlert, setConnected]);

  useEffect(() => {
    connect();
    return () => {
      if (retryRef.current) clearTimeout(retryRef.current);
      wsRef.current?.close();
    };
  }, [connect]);
}
