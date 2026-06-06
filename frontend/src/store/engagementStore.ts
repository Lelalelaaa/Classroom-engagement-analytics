import { create } from 'zustand';
import { LiveMetrics, Alert } from '@/lib/types';

interface EngagementState {
  metrics:     LiveMetrics | null;
  history:     LiveMetrics[];       // Rolling 60-reading window
  alerts:      Alert[];
  isConnected: boolean;
  sessionId:   string | null;

  setMetrics:   (m: LiveMetrics) => void;
  addAlert:     (a: Alert) => void;
  setConnected: (v: boolean) => void;
  setSessionId: (id: string) => void;
  clearSession: () => void;
}

export const useEngagementStore = create<EngagementState>((set) => ({
  metrics:     null,
  history:     [],
  alerts:      [],
  isConnected: false,
  sessionId:   null,

  setMetrics: (m) =>
    set((state) => ({
      metrics: m,
      history: [...state.history.slice(-59), m],
    })),

  addAlert: (a) =>
    set((state) => ({ alerts: [a, ...state.alerts].slice(0, 10) })),

  setConnected: (v) => set({ isConnected: v }),

  setSessionId: (id) => set({ sessionId: id }),

  clearSession: () =>
    set({ metrics: null, history: [], alerts: [], isConnected: false, sessionId: null }),
}));
