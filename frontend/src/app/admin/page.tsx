'use client';
import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { api } from '@/lib/api';
import { ClassroomSummary } from '@/lib/types';
import Link from 'next/link';
import { ArrowLeft, TrendingUp } from 'lucide-react';

export default function AdminOverview() {
  const [classrooms, setClassrooms] = useState<ClassroomSummary[]>([]);
  const [loading, setLoading]       = useState(true);

  useEffect(() => {
    api.getClassroomsSummary()
      .then((data) => setClassrooms(data as ClassroomSummary[]))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const getBarColor = (val: number) =>
    val >= 70 ? '#22c55e' : val >= 45 ? '#f59e0b' : '#ef4444';

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <header className="border-b border-gray-800 px-6 py-4 flex items-center gap-4">
        <Link href="/" className="text-gray-400 hover:text-white transition-colors">
          <ArrowLeft className="w-5 h-5" />
        </Link>
        <h1 className="text-lg font-bold">ReLi — Admin Overview</h1>
        <TrendingUp className="w-5 h-5 text-indigo-400" />
      </header>

      <div className="p-6 space-y-6">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <>
            {/* Bar Chart */}
            <section className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
              <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-6">
                Average Engagement by Classroom
              </h2>
              {classrooms.length === 0 ? (
                <p className="text-gray-500 text-center py-12">No classroom data yet. Start a session to see analytics.</p>
              ) : (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={classrooms} barSize={32}>
                    <XAxis dataKey="classroom_name" stroke="#4b5563" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                    <YAxis domain={[0, 100]} stroke="#4b5563" tick={{ fill: '#9ca3af' }} unit="%" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111827', border: '1px solid #1f2937', borderRadius: 8 }}
                      formatter={(v: number) => [`${v.toFixed(1)}%`, 'Avg Engagement']}
                    />
                    <Bar dataKey="avg_engagement" radius={[6, 6, 0, 0]}>
                      {classrooms.map((c, i) => (
                        <Cell key={i} fill={getBarColor(c.avg_engagement)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </section>

            {/* Classroom Cards */}
            <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {classrooms.map((c) => (
                <div
                  key={c.classroom_name}
                  className="bg-gray-900 border border-gray-800 rounded-2xl p-5 transition-card"
                >
                  <p className="text-sm text-gray-400 font-medium">{c.classroom_name}</p>
                  <p
                    className="text-4xl font-bold mt-2"
                    style={{ color: getBarColor(c.avg_engagement) }}
                  >
                    {c.avg_engagement.toFixed(0)}%
                  </p>
                  <p className="text-xs text-gray-600 mt-1">{c.sessions_today} sessions today</p>
                </div>
              ))}
            </section>
          </>
        )}
      </div>
    </main>
  );
}
