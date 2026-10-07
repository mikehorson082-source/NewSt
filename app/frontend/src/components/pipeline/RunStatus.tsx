import { cn } from 'cn'
import { Circle, CircleCheck, CircleMinus, CirclePause, CircleX, LoaderCircle, TriangleAlert, type LucideIcon } from 'lucide-react'
import { STATUS_LABEL, type RunStatus, type StepStatus } from '@/lib/pipeline'

interface Meta {
  icon: LucideIcon
  className: string
}

// Как и в инвентаре, цвет не несёт смысл один: у каждого статуса свой значок и подпись.
const META: Record<RunStatus | StepStatus, Meta> = {
  running: { icon: LoaderCircle, className: 'animate-spin text-coverage' },
  ok: { icon: CircleCheck, className: 'text-status-good' },
  error: { icon: CircleX, className: 'text-status-critical' },
  skipped: { icon: CircleMinus, className: 'text-status-neutral' },
  stopped: { icon: CirclePause, className: 'text-status-neutral' },
  interrupted: { icon: TriangleAlert, className: 'text-status-serious' },
  pending: { icon: Circle, className: 'text-muted-foreground/60' },
}

export function RunStatusIcon({ status, className }: { status: RunStatus | StepStatus; className?: string }) {
  const { icon: Icon, className: tone } = META[status]
  return <Icon className={cn('size-4 shrink-0', tone, className)} aria-hidden="true" />
}

/** Значок и подпись статуса. */
export function RunStatusLabel({ status, className }: { status: RunStatus | StepStatus; className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-1.5', className)}>
      <RunStatusIcon status={status} />
      {STATUS_LABEL[status]}
    </span>
  )
}
