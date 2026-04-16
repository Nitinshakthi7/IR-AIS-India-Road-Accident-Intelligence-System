"use client";

import React from "react";
import Image from "next/image";

/**
 * FooterIllustration — Static hero-scale footer banner.
 * No infinite framer-motion animations. Raw image presentation.
 * Uses Dashboard.png as the capstone visual.
 */
export function FooterIllustration() {
  return (
    <section className="relative w-full overflow-hidden bg-[#FFFaf5] border-t-2 border-black/10 mt-20">
      {/* Raw image — no opacity modifier, no overlapping gradient mudding */}
      <div className="relative w-full h-[300px] md:h-[420px]">
        <Image
          src="/footer-landscape.png"
          alt="Aerial view of Indian highway interchange with data network"
          fill
          sizes="100vw"
          className="object-cover object-top"
          priority={false}
        />
        {/* Subtle bottom fade only — keeps transition to footer clean */}
        <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-[#FFFaf5] to-transparent z-10 pointer-events-none" />
      </div>
    </section>
  );
}
