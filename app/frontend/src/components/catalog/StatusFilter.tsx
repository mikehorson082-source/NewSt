import { cn } from 'cn'
import { Check, PlusCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Command, CommandGroup, CommandItem, CommandList, CommandSeparator } from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Separator } from '@/components/ui/separator'
import { formatNumber } from '@/lib/format'
import { STATUS, STATUS_ORDER, type Status } from '@/lib/status'
import StatusIcon from '../StatusIcon'

interface Props {
  counts: Record<Status, number> // наборов с каждым статусом во всём инвентаре
  selected: Status[]
  onChange: (statuses: Status[]) => void
}

/** Фильтр «Статус» как в примере shadcn: можно отметить несколько; рядом — число наборов. */
export default function StatusFilter({ counts, selected, onChange }: Props) {
  const statuses = STATUS_ORDER.filter((status) => counts[status] > 0 || selected.includes(status))
  const toggle = (status: Status) =>
    onChange(selected.includes(status) ? selected.filter((item) => item !== status) : [...selected, status])

  return (
    <Popover>
      <PopoverTrigger render={<Button variant="outline" className="border-dashed" />}>
        <PlusCircle />
        Статус
        {selected.length > 0 && (
          <>
            <Separator orientation="vertical" className="mx-1 h-4" />
            <Badge variant="secondary" className="rounded-sm px-1 font-normal lg:hidden">
              {selected.length}
            </Badge>
            <span className="hidden gap-1 lg:flex">
              {selected.length > 2 ? (
                <Badge variant="secondary" className="rounded-sm px-1 font-normal">
                  выбрано {selected.length}
                </Badge>
              ) : (
                STATUS_ORDER.filter((status) => selected.includes(status)).map((status) => (
                  <Badge key={status} variant="secondary" className="rounded-sm px-1 font-normal">
                    {STATUS[status].many}
                  </Badge>
                ))
              )}
            </span>
          </>
        )}
      </PopoverTrigger>
      <PopoverContent className="w-56 p-0" align="start">
        <Command>
          <CommandList>
            <CommandGroup>
              {statuses.map((status) => {
                const checked = selected.includes(status)
                return (
                  <CommandItem key={status} value={status} onSelect={() => toggle(status)}>
                    <span
                      className={cn(
                        'flex size-4 items-center justify-center rounded-[4px] border',
                        checked ? 'border-primary bg-primary text-primary-foreground' : 'border-input [&_svg]:invisible',
                      )}
                    >
                      <Check className="size-3.5 text-primary-foreground" />
                    </span>
                    <StatusIcon status={status} />
                    <span>{STATUS[status].many}</span>
                    <span className="ml-auto font-mono text-xs text-muted-foreground">{formatNumber(counts[status])}</span>
                  </CommandItem>
                )
              })}
            </CommandGroup>
            {selected.length > 0 && (
              <>
                <CommandSeparator />
                <CommandGroup>
                  <CommandItem value="clear" onSelect={() => onChange([])} className="justify-center">
                    Сбросить
                  </CommandItem>
                </CommandGroup>
              </>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
