'use client';
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ReferenceLine, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { LiveMetrics } from '@/lib/types';

interface Props { data: LiveMetrics[] }

export function TimelineLine({ data }: Props) {
  if (data.length === 0) {
    return (
      <div className="h-44 flex items-center justify-center text-gray-600 text-sm">
        Timeline will appear here...
      </div>
    );
  }

  const chartData = data.map((m, i) => ({
    index:      i,
    engagement: m.class_engagement,
    yawn:       m.yawn_rate,
  }));

  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis dataKey="index" hide />
        <YAxis domain={[0, 100]} stroke="#4b5563" tick={{ fill: '#6b7280', fontSize: 11 }} unit="%" />
        <Tooltip
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #1f2937', borderRadius: 8 }}
          formatter={(v: number, name: string) => [`${v.toFixed(1)}%`, name === 'engagement' ? 'Engagement' : 'Yawn Rate']}
        />
        <ReferenceLine y={50} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.5} />
        <Line
          type="monotone" dataKey="engagement"
          stroke="#6366f1" strokeWidth={2} dot={false}
          activeDot={{ r: 4, fill: '#6366f1' }}
        />
        <Line
          type="monotone" dataKey="yawn"
          stroke="#f59e0b" strokeWidth={1.5} dot={false} strokeDasharray="4 4"
          activeDot={{ r: 3, fill: '#f59e0b' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
