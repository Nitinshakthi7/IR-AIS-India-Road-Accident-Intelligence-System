"use client";

import React from "react";
import { cn } from "@/frontend/lib/utils";

/**
 * ScrollPathway — Static dashed SVG road line.
 * Zero JS overhead. Renders once, costs nothing during scroll.
 */
export function ScrollPathway({ className }: { className?: string }) {
  return (
    <div className={cn("absolute inset-0 pointer-events-none z-0 overflow-visible", className)}>
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 100 1000"
        preserveAspectRatio="none"
        className="overflow-visible"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M 50,0 Q 80,250 20,500 T 50,1000"
          className="stroke-black stroke-[0.2] fill-none"
          strokeDasharray="1 2"
          strokeLinecap="round"
          filter="drop-shadow(0.5px 0.5px 0px rgba(107, 196, 179, 1))"
        />
      </svg>
    </div>
  );
}

/**
 * NarrativeScrollPath — Static zig-zag dashed SVG for narrative section.
 * No framer-motion, no scroll tracking, no reflows.
 */
export function NarrativeScrollPath({ className }: { className?: string }) {
  return (
    <div className={cn("absolute inset-0 pointer-events-none z-0", className)}>
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 100 2000"
        preserveAspectRatio="none"
        className="overflow-visible"
      >
        <path
          d="M 50,0 C 50,200 10,300 10,500 S 90,700 90,1000 S 10,1300 10,1500 S 50,1700 50,2000"
          stroke="#1a1a1a"
          strokeWidth="0.2"
          fill="none"
          strokeDasharray="1 2"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}

/**
 * CustomPathway — Static decorative path for use in workspace cards.
 */
export function CustomPathway({ d, className }: { d: string; className?: string }) {
  return (
    <svg
      width="100%"
      height="100%"
      className={cn("overflow-visible absolute inset-0 pointer-events-none", className)}
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d={d}
        className="path-line animate-draw-path"
      />
    </svg>
  );
}
