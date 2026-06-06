import Link from 'next/link';
import { Brain, BarChart3, Shield, Zap } from 'lucide-react';

export default function HomePage() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-gray-950 px-6">
      {/* Hero */}
      <div className="text-center mb-16">
        <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/20 rounded-full px-4 py-1.5 text-indigo-400 text-sm font-medium mb-6">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse-dot" />
          Privacy-Preserving · Real-Time · AI-Powered
        </div>
        <h1 className="text-6xl font-bold tracking-tight text-white mb-4">
          Re<span className="text-indigo-500">Li</span>
        </h1>
        <p className="text-2xl font-light text-gray-400 mb-2">Classroom Engagement Analytics</p>
        <p className="text-gray-500 max-w-lg mx-auto">
          Real-time student engagement monitoring using computer vision.
          No video stored. No faces tracked. Only insights.
        </p>
      </div>

      {/* CTA Buttons */}
      <div className="flex flex-col sm:flex-row gap-4 mb-20">
        <Link
          href="/teacher"
          className="px-8 py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-xl transition-all duration-200 text-center"
        >
          Teacher Dashboard
        </Link>
        <Link
          href="/admin"
          className="px-8 py-3 bg-gray-800 hover:bg-gray-700 text-white font-semibold rounded-xl border border-gray-700 transition-all duration-200 text-center"
        >
          Admin Overview
        </Link>
      </div>

      {/* Feature Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 max-w-4xl w-full">
        {[
          { icon: Brain,    title: 'AI Pipeline',    desc: 'Gaze · Emotion · Yawning — all in-memory' },
          { icon: Shield,   title: 'Privacy-First',  desc: 'Raw video never touches disk or cloud' },
          { icon: Zap,      title: 'Real-Time',      desc: 'Sub-second WebSocket updates to dashboard' },
          { icon: BarChart3, title: 'Insights',       desc: 'Class trends, session history, alerts' },
        ].map(({ icon: Icon, title, desc }) => (
          <div
            key={title}
            className="bg-gray-900 border border-gray-800 rounded-2xl p-5 transition-card"
          >
            <Icon className="w-6 h-6 text-indigo-400 mb-3" />
            <p className="font-semibold text-white mb-1">{title}</p>
            <p className="text-sm text-gray-400">{desc}</p>
          </div>
        ))}
      </div>
    </main>
  );
}
