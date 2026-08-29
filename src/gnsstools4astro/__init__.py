"""GNSStools4Astro — ferramentas para dados GNSS/RINEX aplicados à astronomia.

Pipeline:
    1. download de RINEX (obs + nav) de uma estação GNSS num intervalo de tempo;
    2. extração de satélites e construção de tracks a partir das efemérides
       de transmissão (broadcast ephemeris → propagador Kepler → Skyfield);
    3. estimativa do sinal transmitido: PSD, EIRP e frequências por constelação.

Seguindo o techstack de ``radioskysims`` (DDD, ports/adapter, pydantic, httpx).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
