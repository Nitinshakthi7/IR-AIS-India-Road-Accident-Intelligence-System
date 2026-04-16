"use client";

import React, { useEffect, useRef } from "react";
import Image from "next/image";

const sections = [
  {
    id: "data",
    title: "The Data Harvest",
    subtitle: "PHASE I",
    description: "Distilling meaning from 12,316 historical incident logs. We meticulously cleanse and structure the chaos of the road into high-fidelity intelligence.",
    image: "/data-cleansing.png",
    alignment: "left",
  },
  {
    id: "journey",
    title: "Decoding the Journey",
    subtitle: "PHASE II",
    description: "Every accident is a story of variables. We analyze the terrain, the light, and the human element to map every possible vector of risk.",
    image: "/the-journey.png",
    alignment: "right",
  },
  {
    id: "models",
    title: "The Intelligence Core",
    subtitle: "PHASE III",
    description: "Six specialized machine learning architectures work in tandem, predicting outcomes with 85%+ accuracy to foresee danger before it manifests.",
    image: "/intelligence-core.png",
    alignment: "left",
  },
  {
    id: "safety",
    title: "The Shield of Safety",
    subtitle: "PHASE IV",
    description: "Converting complex mathematics into real-time protection. A sanctuary of predictive safety for every citizen on the Indian road network.",
    image: "/safety-core.png",
    alignment: "right",
  },
];

/**
 * NarrativeRoadPath — Thick flowing S-curve road connecting all 4 narrative cards.
 * Matches the hand-drawn reference: wide organic band snaking from top-right →
 * mid-left → mid-right → bottom-left, like a winding road through the landscape.
 * 100% static SVG — zero JS cost at any time.
 */
function NarrativeRoadPath() {
  return (
    <div className="absolute inset-0 pointer-events-none z-0 overflow-hidden">
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/*
         * Simple dotted line connecting the image cards directly.
         * Card 1 Image (Right) -> Card 2 Image (Left) -> Card 3 (Right) -> Card 4 (Left)
         */}
        <path
          d="M 75,12 C 75,22 25,28 25,38 C 25,48 75,54 75,64 C 75,74 25,80 25,90"
          fill="none"
          stroke="#1a1a1a"
          strokeWidth="0.3"
          strokeDasharray="1.5 2"
          strokeLinecap="round"
          opacity="0.35"
        />
      </svg>
    </div>
  );
}

/**
 * NarrativeCard — Uses a vanilla IntersectionObserver for reveal animation.
 * The observer fires exactly once per card, adds the .is-visible CSS class,
 * then immediately disconnects to free all memory.
 */
function NarrativeCard({ section, idx }: { section: typeof sections[0]; idx: number }) {
  const cardRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const textEl = textRef.current;
    const imgEl = imgRef.current;
    if (!textEl || !imgEl) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            textEl.classList.add("is-visible");
            imgEl.classList.add("is-visible");
            observer.disconnect();
          }
        });
      },
      { threshold: 0.15, rootMargin: "-80px" }
    );

    if (cardRef.current) observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={cardRef}
      className={`flex flex-col ${
        section.alignment === "right" ? "md:flex-row-reverse" : "md:flex-row"
      } items-center gap-12 md:gap-24`}
    >
      {/* Text Area */}
      <div
        ref={textRef}
        className={`flex-1 space-y-6 narrative-card narrative-card-${section.alignment === "left" ? "left" : "right"}`}
        style={{ animationDelay: `${idx * 80}ms` }}
      >
        <div className="fp-sticker bg-white border-black/20 text-black/40">
          {section.subtitle}
        </div>
        <h3 className="editorial-title !text-5xl md:!text-7xl">
          {section.title}
        </h3>
        <p className="text-xl text-black/50 font-medium leading-relaxed max-w-lg">
          {section.description}
        </p>
      </div>

      {/* Image Area */}
      <div
        ref={imgRef}
        className="flex-1 w-full narrative-card narrative-card-scale"
        style={{ animationDelay: `${idx * 80 + 120}ms` }}
      >
        <div className="relative aspect-[4/3] rounded-[2rem] overflow-hidden border-2 border-black shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] bg-white group">
          <Image
            src={section.image}
            alt={section.title}
            fill
            sizes="(max-width: 768px) 100vw, 50vw"
            className="object-contain p-8 opacity-90 group-hover:scale-105 transition-transform duration-700"
          />
          <div className="absolute inset-0 bg-gradient-to-tr from-primary/5 to-transparent pointer-events-none" />
        </div>
      </div>
    </div>
  );
}

export function NarrativeJourney() {
  return (
    <section className="relative py-32 bg-[#FFFaf5] overflow-hidden">
      {/* Thick flowing S-curve road — matching the hand-drawn design */}
      <NarrativeRoadPath />

      <div className="container mx-auto px-6 relative z-10">
        <div className="space-y-[20vh] md:space-y-[30vh]">
          {sections.map((section, idx) => (
            <NarrativeCard key={section.id} section={section} idx={idx} />
          ))}
        </div>
      </div>
    </section>
  );
}


