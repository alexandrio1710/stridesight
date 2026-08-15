import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import ComparisonView from '@/components/ComparisonView';

export const metadata = {
  title: 'Compare Sprints — StrideSight',
};

export default function ComparePage() {
  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="mx-auto max-w-6xl px-6 py-12 sm:py-16">
        <Link
          href="/"
          className="mb-6 inline-flex items-center gap-1.5 text-sm text-neutral-400 transition-colors hover:text-neutral-200"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to single analysis
        </Link>
        <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">Side-by-Side Comparison</h1>
        <p className="mt-1.5 max-w-2xl text-sm text-neutral-400">
          Compare a before/after coaching intervention, or two athletes, with two independent on-device analyses.
        </p>
        <div className="mt-8">
          <ComparisonView />
        </div>
      </div>
    </main>
  );
}
