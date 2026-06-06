'use client';
import { PieChart, Pie, Cell, Legend, Tooltip, ResponsiveContainer } from 'recharts';
import { LiveMetrics } from '@/lib/types';

const EMOTION_COLORS: Record<string, string> = {
  engaged_happy: '#22c55e',
  neutral:       '#60a5fa',
  disengaged:    '#f59e0b',
  distressed:    '#ef4444',
};

const EMOTION_LABELS: Record<string, string> = {
  engaged_happy: 'Engaged & Happy',
  neutral:       'Neutral',
  disengaged:    'Disengaged',
  distressed:    'Distressed',
};

interface Props {
  distribution?: LiveMetrics['emotion_distribution'];
}

export function EmotionPulse({ distribution }: Props) {
  if (!distribution) {
    return (
      <div className="h-44 flex items-center justify-center text-gray-600 text-sm">
        Waiting for data...
      </div>
    );
  }

  const data = Object.entries(distribution)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }));

  if (data.length === 0) {
    return (
      <div className="h-44 flex items-center justify-center text-gray-600 text-sm">
        No faces detected
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={180}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          cx="50%" cy="50%"
          outerRadius={65}
          innerRadius={40}
          paddingAngle={3}
          stroke="none"
        >
          {data.map((entry) => (
            <Cell
              key={entry.name}
              fill={EMOTION_COLORS[entry.name] ?? '#6b7280'}
            />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #1f2937', borderRadius: 8 }}
          formatter={(v: number, name: string) => [v, EMOTION_LABELS[name] ?? name]}
        />
        <Legend
          formatter={(value) => (
            <span className="text-xs text-gray-400">{EMOTION_LABELS[value] ?? value}</span>
          )}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
