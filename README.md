# Analog Pink Noise Generator

> An educational electronics project demonstrating the complete design cycle—from simulation to breadboard implementation—of an analog circuit that generates pink noise with a -3 dB/octave spectral slope.

[![Course](https://img.shields.io/badge/Course-ELEN%203445-blue)](https://github.com/MBane04/AnalogPinkNoiseGenerator)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="images/breadboard/FinalBreadboard.jpg" alt="Final breadboard implementation" width="600"/>
</p>

## 📖 Project Overview

This project demonstrates the practical design and implementation of an analog pink (1/f) noise generator. Pink noise, characterized by equal power per octave, is widely used in audio testing, acoustic measurements, psychoacoustic research, and as a sleep aid. 

The circuit employs a three-stage design:
- **Stage 1:** Generates white noise using the inherent avalanche breakdown noise of a reverse-biased BJT (BC337-16)
- **Stage 2:** Shapes the spectrum using a passive RC shelving filter network to achieve the characteristic -3 dB/octave slope
- **Stage 3:** Amplifies and buffers the signal with adjustable gain control for practical output levels

**Core Specifications:**
*   **Output Noise:** Pink Noise (approx. -3 dB/octave power spectral density slope)
*   **Output Amplitude:** Adjustable up to 2V peak (via potentiometer)
*   **DC Offset:** Centered at 0V
*   **Power Supply:** Dual ±9V DC (from two 9V batteries in series)
*   **Key Components:** BC337-16 BJT, TL072CP Op-Amps, Passive RC Network

## 🎯 Key Features

*   **True Analog Noise Generation:** Leverages avalanche breakdown noise from a reverse-biased BJT for authentic white noise
*   **Passive Frequency Shaping:** Multi-stage RC filter network provides -3 dB/octave spectral slope characteristic of pink noise
*   **Adjustable Output:** User-controllable amplitude (0-2V peak) via potentiometer for flexible applications
*   **Dual Power Supply:** ±9V operation from standard 9V batteries for portability
*   **Comprehensive Validation:** Performance verified through both LTSpice simulation and physical measurements with spectrum analysis
*   **Educational Value:** Demonstrates real-world challenges including component variability, noise source limitations, and the gap between simulation and physical implementation

## 📐 Circuit Schematic

The final circuit schematic is shown below. For detailed design rationale and component selection criteria, refer to the [full project report](docs/PinkNoiseProjectFinalReport.pdf).

<p align="center">
  <img src="images/schematics/Schematic.png" alt="Final Circuit Schematic" width="700"/>
</p>

**Design Highlights:**
*   **Q1 (BC337-16):** Operated in reverse breakdown mode as the white noise source, biased by the -9V rail
*   **U1A (TL072):** First-stage amplifier providing initial gain (~60 dB) to boost the weak noise signal
*   **R7, R8, C5, C6, C7:** Passive RC filter network forming three cascaded shelving stages to approximate the -3 dB/octave pink noise spectrum
*   **U2A (TL072):** Output buffer stage with user-adjustable gain via potentiometer R9 for amplitude control

## 📦 Bill of Materials (BOM)

| Qty | Component | Value / Part Number | Notes |
| :-- | :--- | :--- | :--- |
| 1 | BJT Transistor | BC337-16 | White noise source |
| 2 | Op-Amp IC | TL072CP | Dual JFET-input operational amplifier |
| 2 | Battery | 9V | Power supply |
| 1 | Capacitor | 100μF Electrolytic | Power rail decoupling |
| 2 | Capacitor | 10μF | Coupling/Decoupling |
| 1 | Capacitor | 10μF Non-Polarized | Filter network |
| 1 | Capacitor | 820pF | Filter network |
| 1 | Capacitor | 2.2nF | Filter network |
| 1 | Capacitor | 5.6nF | Filter network |
| 8 | Resistors (1%) | Various (1MΩ, 100kΩ, 18kΩ, etc.) | See schematic for values |
| 1 | Potentiometer | 100kΩ Linear | Output amplitude adjustment |
| 1 | Piezoelectric Speaker | - | For audible output |
| 1 | Breadboard | - | For prototyping |
| - | Wire | - | For connections |

*(The complete cost breakdown is available in the project report.)*

## 🔧 Build & Test Instructions

### 1. Circuit Simulation (LTSpice)
*   Simulation files and analysis scripts will be placed in [simulation/](simulation) and depend on LTSpice outputs; links will be added here once committed
*   **Important Note:** LTSpice cannot accurately model avalanche breakdown noise in BJTs. As a workaround, simulations use either:
    - A pre-recorded `.wave` file of white noise, or
    - The `WHITE()` voltage source function
*   Run the `.asc` schematic files to perform transient, AC, and noise analyses
*   This limitation highlights an important lesson: some real-world phenomena cannot be fully captured by simulation tools

### 2. Physical Assembly
1.  **Component Placement:** Position the ICs (TL072CP) and transistor (BC337-16) on the breadboard, ensuring correct pin orientation (check datasheets)
2.  **Power Distribution:** Connect the dual power supply (+9V and -9V) from series-connected batteries to the breadboard rails. Add decoupling capacitors (100μF electrolytic) close to each IC to minimize noise
3.  **Noise Source Stage:** Assemble the white noise generator around Q1, with proper reverse-bias configuration
4.  **Filter and Amplifier Stages:** Build the RC filter network and op-amp stages around U1 and U2 according to the schematic
5.  **Output Stage:** Connect the potentiometer for gain adjustment and the speaker/output jack for monitoring

### 3. Testing & Validation
1.  **Power Supply Verification:** Use a multimeter to confirm correct voltages at all critical nodes before proceeding
2.  **Waveform Inspection:** Connect an oscilloscope to the output. You should observe a chaotic, random signal with approximately 0V DC offset and the expected amplitude range
3.  **Spectral Analysis (Critical Validation):**
    *   Connect the circuit output to a spectrum analyzer, or alternatively use a PC sound card with FFT analysis software (e.g., Audacity, MATLAB, Python with scipy)
    *   Analyze the Power Spectral Density (PSD) across the audio frequency range (20 Hz - 20 kHz)
    *   **Success Criteria:** The PSD should exhibit an approximately -3 dB/octave slope (equivalent to -10 dB/decade), characteristic of pink noise
    *   Compare measured results with simulation predictions to evaluate design accuracy

## 📈 Results & Performance

The circuit successfully generated pink noise and demonstrated the practical challenges of analog circuit design. Below is a comprehensive comparison between simulated predictions and measured performance.

### Key Findings

**Challenges Encountered:**
*   The BC337-16 BJTs exhibited lower-than-expected avalanche breakdown noise amplitude, necessitating a design modification
*   Solution: Increased first-stage gain by replacing the 100kΩ feedback resistor with 10MΩ (100× gain increase)
*   Trade-off: While this achieved the required output voltage, it resulted in a spectrum slightly "whiter" than ideal pink noise

**Performance Metrics:**
*   **Spectral Slope:** Achieved approximately -3 dB/octave across most of the audio band
*   **Output Amplitude:** Successfully met the 2V peak specification with adjustable control
*   **Project Score:** 145.16 (normalized error metric, where lower values indicate closer adherence to ideal pink noise characteristics)

### 📊 Simulation vs. Real Results Comparison

<div align="center">

<table>
<tr>
<th>🖥️ Simulation (LTSpice)</th>
<th>🔌 Real Circuit (Breadboard)</th>
</tr>
<tr>
<td align="center">
<img src="images/simresults/PSDDataandSampledModel.png" alt="Simulated PSD" width="450"/><br/>
<em>Simulated PSD with Model</em>
</td>
<td align="center">
<img src="images/realresults/PSDDataandSampledModel.png" alt="Measured PSD" width="450"/><br/>
<em>Measured PSD with Model</em>
</td>
</tr>
<tr>
<td align="center">
<img src="images/simresults/PSDLinearFit.png" alt="Simulated Linear Fit" width="450"/><br/>
<em>Simulated PSD Linear Fit</em>
</td>
<td align="center">
<img src="images/realresults/PSDLinearFit.png" alt="Measured Linear Fit" width="450"/><br/>
<em>Measured PSD Linear Fit</em>
</td>
</tr>
<tr>
<td align="center">
<img src="images/simresults/VoltageTimeGraph.png" alt="Simulated Voltage" width="450"/><br/>
<em>Simulated Voltage vs. Time</em>
</td>
<td align="center">
<img src="images/realresults/VoltageTimeGraph.png" alt="Measured Voltage" width="450"/><br/>
<em>Measured Voltage vs. Time</em>
</td>
</tr>
<tr>
<td align="center">
<img src="images/simresults/Histogram.png" alt="Simulated Histogram" width="450"/><br/>
<em>Simulated Distribution</em>
</td>
<td align="center">
<img src="images/realresults/Histogram.png" alt="Measured Histogram" width="450"/><br/>
<em>Measured Distribution</em>
</td>
</tr>
</table>

</div>

### 🛠️ Build Progress

<p align="center">
  <img src="images/breadboard/PrototypeBreadboard.jpg" alt="Prototype Build" width="500"/>
  <br/>
  <em>Early Prototype on Breadboard</em>
</p>

<p align="center">
  <img src="images/breadboard/FinalBreadboard.jpg" alt="Final Build" width="500"/>
  <br/>
  <em>Final Breadboard Implementation</em>
</p>

## 📚 Documentation

**Complete Project Report:** [`PinkNoiseProjectFinalReport.pdf`](docs/PinkNoiseProjectFinalReport.pdf)

The comprehensive 15-page report includes:
*   Theoretical background on pink noise and 1/f noise characteristics
*   Detailed design methodology and component selection rationale  
*   Complete simulation results and analysis
*   Physical implementation challenges and solutions
*   Quantitative performance evaluation and error analysis
*   Comparison between theoretical, simulated, and measured results

## 🔊 Audio Samples

Explore and listen to recorded signals in [audio/](audio/):
- **Ideal Pink Noise:** Reference signal for spectral comparison.
- **Simulated Pink Noise:** Pink noise generated via simulation workflow.
- **Python White Noise:** Synthetic white-noise source used for testing/simulation.

<div align="center">
  <p><strong>Ideal Pink Noise</strong><br/>
    <audio controls>
      <source src="https://raw.githubusercontent.com/MBane04/AnalogPinkNoiseGenerator/main/audio/idealPinkNoise.wav" type="audio/wav"/>
      Your browser does not support the audio element.
    </audio>
  </p>
  <p><strong>Simulated Pink Noise</strong><br/>
    <audio controls>
      <source src="https://raw.githubusercontent.com/MBane04/AnalogPinkNoiseGenerator/main/audio/simulatedPinkNoise.wav" type="audio/wav"/>
      Your browser does not support the audio element.
    </audio>
  </p>
  <p><strong>Python White Noise</strong><br/>
    <audio controls>
      <source src="https://raw.githubusercontent.com/MBane04/AnalogPinkNoiseGenerator/main/audio/whiteNoise.wav" type="audio/wav"/>
      Your browser does not support the audio element.
    </audio>
  </p>
</div>


## 🔮 Future Improvements

While this project successfully demonstrates the core concepts of analog pink noise generation, several enhancements could improve performance:

1.  **Enhanced Noise Source:** 
    - Evaluate Zener diodes as alternative noise sources
    - Screen/select BJTs with higher avalanche noise output
    - Consider specialized noise diode ICs for consistent performance

2.  **Optimized Filter Design:**
    - Implement additional shelving stages for closer approximation to ideal -3 dB/octave slope
    - Explore active filter topologies for better spectral shaping

3.  **PCB Implementation:**
    - Design a custom printed circuit board to reduce parasitic capacitance and improve noise immunity
    - Implement proper grounding techniques and power plane design

4.  **Power Supply Enhancement:**
    - Replace batteries with a regulated ±9V or ±12V supply for consistent long-term operation
    - Add voltage regulation and additional filtering to minimize power supply noise

  ## 📝 TODOs

  - [ ] **LTSpice Schematics:** Add `.asc` files to [simulation/](simulation) and link them under “Circuit Simulation (LTSpice)”.
  - [ ] **Python Analysis Scripts:** Add PSD estimation, slope fitting, histogram, and time-domain plotting scripts to [simulation/](simulation); include usage and quick-start commands.
  - [ ] **Figure Mapping:** Add captions or notes tying README images to report figures (e.g., “Fig. 3–7”) for easier cross-reference.
  - [ ] **BOM Cross-Check:** Verify component values vs. schematic and add any missing specifics.

## 👥 Contributors

**Project Team:**
*   **Mason Bane** - Circuit design, simulation, breadboard implementation, and testing
*   **Cameron Osborne** - Circuit design, simulation, breadboard implementation, and testing

**Acknowledgments:**
*   **Dr. Eric J. Wyers** - Course instructor, project advisor, and technical guidance
*   **ELEN 3445: Electronics 1** - University of Denver, Department of Electrical and Computer Engineering

## 📄 License

This hardware design and documentation are shared under the **MIT License**. See the `LICENSE` file for details.