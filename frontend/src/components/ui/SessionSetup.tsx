'use client';
import { useState, useEffect } from 'react';
import { api } from '@/lib/api';
import { Classroom } from '@/lib/types';
import { Brain, Loader2, Zap } from 'lucide-react';
import Link from 'next/link';

// Demo classrooms shown when backend is offline
const DEMO_CLASSROOMS: Classroom[] = [
  { id: 'demo-1', name: 'Room 101 — Computer Science', school_name: 'Demo School', capacity: 30 },
  { id: 'demo-2', name: 'Room 202 — Mathematics',      school_name: 'Demo School', capacity: 25 },
  { id: 'demo-3', name: 'Room 305 — Physics',          school_name: 'Demo School', capacity: 28 },
];

interface Props {
  onStart:  (classroomId: string, teacher: string, subject: string) => void;
  onDemo?:  () => void;
  loading:  boolean;
}

export function SessionSetup({ onStart, onDemo, loading }: Props) {
  const [classrooms, setClassrooms]   = useState<Classroom[]>(DEMO_CLASSROOMS);
  const [classroomId, setClassroomId] = useState(DEMO_CLASSROOMS[0].id);
  const [teacher, setTeacher]         = useState('');
  const [subject, setSubject]         = useState('');
  const [offline, setOffline]         = useState(false);

  useEffect(() => {
    api.listClassrooms().then((data) => {
      const list = data as Classroom[];
      if (list.length > 0) {
        setClassrooms(list);
        setClassroomId(list[0].id);
        setOffline(false);
      } else {
        setOffline(true);
      }
    }).catch(() => {
      setOffline(true);
      setClassrooms(DEMO_CLASSROOMS);
      setClassroomId(DEMO_CLASSROOMS[0].id);
    });
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!classroomId) return;
    onStart(classroomId, teacher, subject);
  };

  return (
    <main className="min-h-screen bg-gray-950 flex items-center justify-center px-6">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Brain className="w-10 h-10 text-indigo-400 mx-auto mb-3" />
          <h1 className="text-2xl font-bold text-white">Start a Session</h1>
          <p className="text-gray-400 text-sm mt-1">Configure the monitoring session for your classroom</p>
        </div>

        {offline && (
          <div className="mb-4 flex items-center gap-2 bg-yellow-500/10 border border-yellow-500/20 rounded-xl px-4 py-3 text-yellow-400 text-xs">
            <span className="w-2 h-2 rounded-full bg-yellow-400 shrink-0" />
            Backend offline — showing demo classrooms. Live AI requires the backend.
          </div>
        )}

        <form onSubmit={handleSubmit} className="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1.5 uppercase tracking-wider">Classroom</label>
            <select
              value={classroomId}
              onChange={(e) => setClassroomId(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-500"
              required
            >
              {classrooms.map((c) => (
                <option key={c.id} value={c.id}>{c.name} ({c.school_name})</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1.5 uppercase tracking-wider">Teacher Name</label>
            <input
              type="text"
              value={teacher}
              onChange={(e) => setTeacher(e.target.value)}
              placeholder="e.g. Ms. Nguyen"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-500"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1.5 uppercase tracking-wider">Subject</label>
            <input
              type="text"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="e.g. Mathematics"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-indigo-500"
            />
          </div>

          <div className="flex gap-3 pt-1">
            {/* Demo Mode button — always available */}
            {onDemo && (
              <button
                type="button"
                onClick={onDemo}
                className="flex-1 py-2.5 bg-indigo-500/10 hover:bg-indigo-500/20 border border-indigo-500/30 text-indigo-400 font-semibold rounded-lg transition-colors flex items-center justify-center gap-2 text-sm"
              >
                <Zap className="w-4 h-4" />
                Demo Mode
              </button>
            )}
            <button
              type="submit"
              disabled={loading || !classroomId}
              className="flex-1 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-colors flex items-center justify-center gap-2 text-sm"
            >
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {loading ? 'Starting...' : 'Start Session'}
            </button>
          </div>
        </form>

        <p className="text-center text-xs text-gray-600 mt-4">
          <Link href="/admin" className="hover:text-gray-400 transition-colors">Admin Overview</Link>
        </p>
      </div>
    </main>
  );
}
