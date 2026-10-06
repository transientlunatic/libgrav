//! Puddin — mathematical and physical primitives for gravitational-wave astronomy.
//!
//! This crate provides the geometric and algebraic building blocks needed to
//! describe compact binary systems, coordinate systems, and sky geometry.
//! It is deliberately narrow in scope: no waveform models, no detector noise
//! curves, no data analysis.  Instead it supplies the routines that every such
//! package needs but should not have to reimplement.
//!
//! # Feature flags
//! None at present.  All modules are unconditionally compiled.

pub mod binary;
