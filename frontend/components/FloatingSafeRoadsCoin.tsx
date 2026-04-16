"use client";

import React from "react";
import { ShieldCheck } from "lucide-react";

/**
 * FloatingSafeRoadsCoin — Lightweight fixed coin badge.
 * Uses pure CSS animate-bounce instead of framer-motion scroll physics.
 * Only renders when isVisible=true (home tab only).
 */
export function FloatingSafeRoadsCoin({ isVisible = true }: { isVisible?: boolean }) {
  if (!isVisible) return null;

  return (
    <div className="fixed bottom-10 right-10 z-[100] hidden lg:flex flex-col items-center gap-2 pointer-events-none animate-bounce-slow">
      <div className="h-20 w-20 rounded-full bg-primary border-4 border-black shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] flex items-center justify-center">
        <ShieldCheck className="h-10 w-10 text-black" />
      </div>
      <div className="bg-black text-[10px] text-white px-3 py-1 rounded-full font-bold uppercase tracking-widest border-2 border-black shadow-[3px_3px_0px_0px_rgba(107,196,179,1)]">
        Safe Paths
      </div>
    </div>
  );
}
