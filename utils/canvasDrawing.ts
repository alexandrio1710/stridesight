/**
 * StrideSight — Canvas Drawing Helpers
 *
 * Pure functions of (ctx, data) that render the skeleton overlay and
 * color-coded joint-angle arcs onto a 2D canvas. Split out of the video
 * analysis hook so it stays a plain rendering layer with no React/MediaPipe
 * dependencies of its own — anything that has a CanvasRenderingContext2D and
 * a frame's landmarks/metrics can call these.
 */

import {
  getSideJoints,
  isLandmarkReliable,
  POSE_LANDMARK_INDICES,
  type FrameMetrics,
  type Landmark,
  type MetricStatus,
  type PoseLandmarks,
} from './biomechanics';

/** [startLandmarkIndex, endLandmarkIndex] pair, as exported by @mediapipe/pose's POSE_CONNECTIONS. */
export type Connection = readonly [number, number];

export function statusToRGBA(status: MetricStatus | null, alpha: number): string {
  switch (status) {
    case 'optimal':
      return `rgba(74, 222, 128, ${alpha})`; // green-400
    case 'caution':
      return `rgba(250, 204, 21, ${alpha})`; // yellow-400
    case 'suboptimal':
      return `rgba(248, 113, 113, ${alpha})`; // red-400
    default:
      return `rgba(148, 163, 184, ${alpha})`; // slate-400
  }
}

export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: PoseLandmarks,
  connections: readonly Connection[],
  width: number,
  height: number
): void {
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)'; // sky-400

  connections.forEach(([startIdx, endIdx]) => {
    const start = landmarks[startIdx];
    const end = landmarks[endIdx];
    if (!isLandmarkReliable(start) || !isLandmarkReliable(end)) return;

    ctx.beginPath();
    ctx.moveTo(start.x * width, start.y * height);
    ctx.lineTo(end.x * width, end.y * height);
    ctx.stroke();
  });

  ctx.fillStyle = 'rgba(224, 242, 254, 0.95)'; // sky-100
  landmarks.forEach((landmark) => {
    if (!isLandmarkReliable(landmark)) return;
    ctx.beginPath();
    ctx.arc(landmark.x * width, landmark.y * height, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

/**
 * Draws a semi-transparent pie-slice arc at `vertex`, spanning the interior
 * angle between rays vertex->a and vertex->c. The sweep direction is chosen
 * so the arc always fills the *shorter* (interior) angle rather than the
 * reflex angle, matching the value produced by calculateAngle().
 */
export function drawJointArc(
  ctx: CanvasRenderingContext2D,
  a: Landmark,
  vertex: Landmark,
  c: Landmark,
  width: number,
  height: number,
  color: string,
  radius = 32
): void {
  const pVertex = { x: vertex.x * width, y: vertex.y * height };
  const pA = { x: a.x * width, y: a.y * height };
  const pC = { x: c.x * width, y: c.y * height };

  const angleA = Math.atan2(pA.y - pVertex.y, pA.x - pVertex.x);
  const angleC = Math.atan2(pC.y - pVertex.y, pC.x - pVertex.x);

  let delta = angleC - angleA;
  while (delta > Math.PI) delta -= 2 * Math.PI;
  while (delta < -Math.PI) delta += 2 * Math.PI;

  const anticlockwise = delta < 0;

  ctx.beginPath();
  ctx.moveTo(pVertex.x, pVertex.y);
  ctx.arc(pVertex.x, pVertex.y, radius, angleA, angleC, anticlockwise);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();

  ctx.lineWidth = 1.5;
  ctx.strokeStyle = color.replace(/, [\d.]+\)$/, ', 0.9)');
  ctx.stroke();
}

/**
 * Draws the trunk-lean arc at the hip, between the torso line and a vertical
 * reference — the visual counterpart to the trunk lean angle that drives
 * phase detection. Uses shoulder/hip midpoints computed here (not exported
 * from the biomechanics module, which only needs the final angle).
 */
export function drawTrunkLeanArc(
  ctx: CanvasRenderingContext2D,
  landmarks: PoseLandmarks,
  metrics: FrameMetrics,
  width: number,
  height: number
): void {
  if (metrics.trunkLeanAngle === null) return;

  const idx = POSE_LANDMARK_INDICES;
  const leftShoulder = landmarks[idx.LEFT_SHOULDER];
  const rightShoulder = landmarks[idx.RIGHT_SHOULDER];
  const leftHip = landmarks[idx.LEFT_HIP];
  const rightHip = landmarks[idx.RIGHT_HIP];

  if (
    !isLandmarkReliable(leftShoulder) ||
    !isLandmarkReliable(rightShoulder) ||
    !isLandmarkReliable(leftHip) ||
    !isLandmarkReliable(rightHip)
  ) {
    return;
  }

  const shoulderMid: Landmark = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
    z: 0,
  };
  const hipMid: Landmark = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2,
    z: 0,
  };
  const verticalReference: Landmark = { x: hipMid.x, y: hipMid.y - 0.15, z: 0 };

  drawJointArc(
    ctx,
    shoulderMid,
    hipMid,
    verticalReference,
    width,
    height,
    statusToRGBA(metrics.trunkLeanStatus, 0.3),
    46
  );
}

export function drawAngleArcs(
  ctx: CanvasRenderingContext2D,
  landmarks: PoseLandmarks,
  metrics: FrameMetrics,
  width: number,
  height: number
): void {
  drawTrunkLeanArc(ctx, landmarks, metrics, width, height);

  if (metrics.kneeDriveAngle !== null && metrics.kneeDriveSide) {
    const { hip, knee, ankle } = getSideJoints(landmarks, metrics.kneeDriveSide);
    drawJointArc(ctx, hip, knee, ankle, width, height, statusToRGBA(metrics.kneeDriveStatus, 0.35));
  }

  if (metrics.hipExtensionAngle !== null && metrics.hipExtensionSide) {
    const { shoulder, hip, knee } = getSideJoints(landmarks, metrics.hipExtensionSide);
    drawJointArc(ctx, shoulder, hip, knee, width, height, statusToRGBA(metrics.hipExtensionStatus, 0.35));
  }

  if (metrics.leftArmSwingAngle !== null) {
    const { shoulder, elbow, wrist } = getSideJoints(landmarks, 'left');
    drawJointArc(ctx, shoulder, elbow, wrist, width, height, statusToRGBA(metrics.leftArmSwingStatus, 0.35), 24);
  }

  if (metrics.rightArmSwingAngle !== null) {
    const { shoulder, elbow, wrist } = getSideJoints(landmarks, 'right');
    drawJointArc(ctx, shoulder, elbow, wrist, width, height, statusToRGBA(metrics.rightArmSwingStatus, 0.35), 24);
  }
}
