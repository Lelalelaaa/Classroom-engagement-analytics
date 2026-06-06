'use client';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

interface Props { value: number }

export function EngagementGauge({ value }: Props) {
  const clamped = Math.min(100, Math.max(0, value));
  const color   = clamped >= 70 ? '#22c55e' : clamped >= 45 ? '#f59e0b' : '#ef4444';
  const data    = [
    { name: 'engaged', value: clamped },
    { name: 'gap',     value: 100 - clamped },
  ];

  return (
    <div className="relative flex items-center justify-center" style={{ height: 180 }}>
      <ResponsiveContainer width="100%" height={180}>
        <PieChart>
          <Pie
            data={data}
            cx="50%" cy="80%"
            startAngle={180} endAngle={0}
            innerRadius={65} outerRadius={80}
            dataKey="value"
            stroke="none"
            paddingAngle={0}
          >
            <Cell fill={color} />
            <Cell fill="#1f2937" />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute bottom-4 text-center">
        <p className="text-4xl font-bold" style={{ color }}>{clamped.toFixed(0)}%</p>
        <p className="text-xs text-gray-500 mt-1">Class Engagement</p>
      </div>
    </div>
  );
}
