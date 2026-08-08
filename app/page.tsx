import { Activity } from 'lucide-react';
import VideoAnalyzer from '@/components/VideoAnalyzer';

export default function Home() {
  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="mx-auto max-w-6xl px-6 py-12 sm:py-16">
        <header className="mb-10 flex flex-col gap-4 border-b border-white/10 pb-8">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-sky-500/10 ring-1 ring-sky-400/30">
              <Activity className="h-5 w-5 text-sky-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
                Stride<span className="text-sky-400">Sight</span>
              </h1>
              <p className="text-sm text-neutral-400">AI-Powered Sprint Biomechanics Analysis</p>
            </div>
          </div>

          <p className="max-w-3xl text-sm leading-relaxed text-neutral-400 sm:text-base">
            StrideSight runs on-device computer vision (MediaPipe Pose) over sprint footage to
            extract frame-by-frame joint kinematics — knee drive, hip extension, and arm swing are
            computed from real-world 3D landmarks via vector dot products, which removes the
            foreshortening error a flat 2D projection introduces whenever a limb moves toward or
            away from the camera. Trunk lean is measured in the 2D image plane on purpose, against
            true vertical, since MediaPipe&apos;s 3D space has no documented gravity reference to
            measure against. Acceleration and max-velocity running are biomechanically different
            gaits, so StrideSight detects which phase you are in from your own posture and scores
            each phase against its own literature-informed targets, rather than judging your whole
            sprint against one fixed standard. Every frame is processed locally in your browser; no
            video is ever uploaded to a server.
          </p>

          <div className="flex flex-wrap gap-2 pt-1">
            {[
              'MediaPipe Pose',
              '3D Vector Kinematics',
              'Phase-Aware Scoring',
              'Real-Time Canvas Overlay',
              'CSV Export',
            ].map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-neutral-300"
              >
                {tag}
              </span>
            ))}
          </div>
        </header>

        <VideoAnalyzer />

        <footer className="mt-16 border-t border-white/10 pt-6 text-center text-xs text-neutral-500">
          StrideSight — a computer vision &amp; biomechanics research project.
        </footer>
      </div>
    </main>
  );
}
