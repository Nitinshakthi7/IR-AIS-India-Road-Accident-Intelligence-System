"use client";

import React from "react";
import Image from "next/image";
import { Activity, ArrowRight, MousePointerClick } from "lucide-react";

export function LandingHero() {
  const scrollToContent = () => {
    const main = document.querySelector("main");
    if (main) {
      main.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <section className="relative w-full bg-[#FFFaf5] overflow-hidden">
      {/*
       * Text block: absolutely positioned so it hovers in the empty "sky"
       * portion of the hero image, clamped directly below the Navbar.
       */}
      <div className="absolute top-0 left-0 right-0 z-10 pt-10 pb-6 px-6">
        <div className="container mx-auto">
          <div className="flex flex-col items-center text-center max-w-4xl mx-auto">
            <div className="fp-sticker mb-6 animate-fade-in-up">
              <Activity className="h-3 w-3 text-primary" />
              <span>National Traffic Intelligence Bureau</span>
            </div>

            <h1
              className="editorial-title text-5xl md:text-8xl mb-6 animate-fade-in-up"
              style={{ animationDelay: "0.1s" }}
            >
              Predicting the <br />
              <span className="text-primary italic">Unpredictable</span>.
            </h1>

            <p
              className="text-lg md:text-xl text-black/60 font-medium mb-8 max-w-2xl leading-relaxed animate-fade-in-up"
              style={{ animationDelay: "0.2s" }}
            >
              A sophisticated machine learning engine deciphering 12,316 accident records to build a
              safer, more intelligent road network for India.
            </p>

            <div
              className="flex flex-wrap items-center justify-center gap-6 animate-fade-in-up"
              style={{ animationDelay: "0.3s" }}
            >
              <button
                onClick={scrollToContent}
                className="fp-button fp-button-primary scale-110 !px-8 !py-4"
              >
                Enter Dashboard
                <ArrowRight className="h-5 w-5" />
              </button>
              <div className="flex items-center gap-4 text-sm font-bold uppercase tracking-widest text-black/40">
                <MousePointerClick className="h-4 w-4" />
                <span>Scroll to explore</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/*
       * Hero image in normal flow — width + h-auto dictates section height
       * based on the image's natural aspect ratio. Nothing gets cropped.
       */}
      <Image
        src="/hero-illustration.png"
        alt="Etched illustration of an Indian highway system with data-network overlays"
        width={1600}
        height={900}
        sizes="100vw"
        className="w-full h-auto object-contain opacity-90"
        priority
      />

      {/* Soft bottom blend into next section */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-[#FFFaf5] to-transparent pointer-events-none" />
    </section>
  );
}
