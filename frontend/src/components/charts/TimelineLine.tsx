'use client';
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ReferenceLine, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { LiveMetrics } from '@/lib/types';

type TimelineDataPoint = LiveMetrics | { time: number; engagement: number };

interface Props { data: TimelineDataPoint[] }

export function TimelineLine({ data }: Props) {
  if (data.length === 0) {
    return (
      <div className="h-44 flex items-center justify-center text-gray-600 text-sm">
        Timeline will appear here...
      </div>
    );
  }

  const isLiveMetrics = (d: TimelineDataPoint): d is LiveMetrics =>
    'class_engagement' in d;

  const chartData = data.map((m, i) => ({
    index:      i,
    engagement: isLiveMetrics(m) ? m.class_engagement : m.engagement,
  }));

  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis dataKey="index" hide />
        <YAxis domain={[0, 100]} stroke="#4b5563" tick={{ fill: '#6b7280', fontSize: 11 }} unit="%" />
        <Tooltip
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #1f2937', borderRadius: 8 }}
          formatter={(v: number) => [`${v.toFixed(1)}%`, 'Engagement']}
        />
        <ReferenceLine y={50} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.5} />
        <Line
          type="monotone" dataKey="engagement"
          stroke="#6366f1" strokeWidth={2} dot={false}
          activeDot={{ r: 4, fill: '#6366f1' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
