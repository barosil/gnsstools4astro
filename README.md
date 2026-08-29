# gnsstools4astro

Ferramentas para dados GNSS/RINEX aplicados à astronomia.

## Pipeline

1. download de RINEX (obs + nav) de uma estação GNSS num intervalo de tempo;
2. extração de satélites e construção de tracks a partir das efemérides
   de transmissão (broadcast ephemeris → propagador Kepler → Skyfield);
3. estimativa do sinal transmitido: PSD, EIRP e frequências por constelação.

Seguindo o techstack de `radioskysims` (DDD, ports/adapter, pydantic, httpx).

## Integração opcional com radioskysims

`infrastructure/ephemeris/tle_tracks.py` usa (lazy/opcional) o banco e o
download de satélites dos pacotes `radioskysims-core` / `radioskysims-sources`
quando disponíveis; degrada com `None`/`RuntimeError` caso contrário.

## Testes

```bash
uv run --package gnsstools4astro pytest
```
