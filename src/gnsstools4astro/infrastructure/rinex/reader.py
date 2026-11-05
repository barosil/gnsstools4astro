"""Adaptador para leitura de RINEX via georinex.

Mapeia os datasets de navegação (NAV) para modelos de efemérides do domínio:

* GPS/GALILEO/BEIDOU → :class:`~gnsstools4astro.domain.satellite.BroadcastEphemeris`
  (elementos Kepler, ICD-GPS-200);
* GLONASS → :class:`~gnsstools4astro.domain.satellite.GlonassEphemeris`
  (vetor de estado + aceleração).

E extrai amostras de C/N0 dos arquivos de observação (OBS).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import georinex as gr
import numpy as np
import xarray as xr
from gnsstools4astro.domain.satellite import (
    BroadcastEphemeris,
    Constellation,
    GlonassEphemeris,
    constellation_from_sv,
)
from gnsstools4astro.domain.signal import SignalSample
from gnsstools4astro.infrastructure.rinex.nav2 import is_rinex2_nav, read_nav2

_GPS_EPOCH = datetime(1980, 1, 6, tzinfo=UTC)
_BDS_EPOCH = datetime(2006, 1, 1, tzinfo=UTC)  # época da semana BDS (Beidou)
_WEEK_S = 7 * 86400.0

# Campo de semana por constelação (RINEX 3).
_WEEK_FIELD = {
    Constellation.GPS: "GPSWeek",
    Constellation.GALILEO: "GALWeek",
    Constellation.BEIDOU: "BDTWeek",
}


def _gps_week_to_datetime(week: float, sec_of_week: float) -> datetime:
    return _GPS_EPOCH + timedelta(weeks=int(week)) + timedelta(seconds=float(sec_of_week))


def _bds_week_to_datetime(week: float, sec_of_week: float) -> datetime:
    """Semana BDS (Beidou) tem época própria: 2006-01-01 00:00 UTC."""
    return _BDS_EPOCH + timedelta(weeks=int(week)) + timedelta(seconds=float(sec_of_week))


def _canonical_sv(sv: str) -> str:
    """Remove o sufixo ``_N`` que o georinex adiciona a SV duplicados."""
    return re.sub(r"_\d+$", "", sv)


def _to_datetime(value: object) -> datetime:
    """Converte numpy datetime64 (ou datetime) para datetime aware (UTC)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    arr = np.asarray(value)
    if arr.dtype.kind == "M":
        ts = arr.astype("datetime64[ns]").astype("int64") / 1e9
        return datetime.fromtimestamp(ts.item(), tz=UTC)
    msg = f"tipo de tempo não suportado: {type(value)}"
    raise TypeError(msg)


class RinexReader:
    """Lê arquivos RINEX (obs/nav) com georinex e mapeia para modelos."""

    def load(self, path: str | Path) -> xr.Dataset:
        """Carrega um arquivo RINEX com georinex."""
        return gr.load(str(path))

    def load_obs(self, path: str | Path) -> xr.Dataset:
        """Carrega um arquivo de observação."""
        return self.load(path)

    def load_nav(self, path: str | Path) -> xr.Dataset:
        """Carrega um arquivo de navegação.

        * **NAV RINEX 2.11** → parser próprio
          (:mod:`gnsstools4astro.infrastructure.rinex.nav2`), que mantém o
          último registro em épocas duplicadas (o ``georinex.nav2`` descarta o
          SV inteiro e emite avisos "duplicate times detected");
        * **NAV RINEX 3** (ex.: ``BRDC00IGS_R`` do CDDIS) → georinex com
          ``use="GREC"``, carregando apenas GPS/GLONASS/GALILEO/BEIDOU (o
          combinado do IGS também traz IRNSS/QZSS, que o georinex não
          consegue mapear).
        """
        if is_rinex2_nav(path):
            return read_nav2(path)
        return gr.load(str(path), use="GREC")

    # ------------------------------------------------------------------
    # Navegação → efemérides
    # ------------------------------------------------------------------

    def to_ephemerides(self, nav: xr.Dataset) -> list[BroadcastEphemeris]:
        """Converte NAV (GPS/GALILEO/BEIDOU) em efemérides Kepler.

        Para cada SV considera todas as mensagens válidas do arquivo (por
        época). O SV é normalizado (remove o sufixo ``_N`` do georinex), e a
        seleção da melhor mensagem por janela fica a cargo do chamador.
        Registros GLONASS são ignorados aqui (ver :meth:`to_glonass_ephemerides`).
        """
        if "sv" not in nav.coords:
            return []
        ephs: list[BroadcastEphemeris] = []
        for sv_raw in nav["sv"].values:
            sv = str(sv_raw)
            try:
                constellation = constellation_from_sv(sv)
            except ValueError:
                continue
            if constellation is Constellation.GLONASS:
                continue
            ephs.extend(self._sv_kepler_series(nav, sv))
        return ephs

    def _sv_kepler_series(
        self,
        nav: xr.Dataset,
        sv: str,
    ) -> list[BroadcastEphemeris]:
        """Todas as efemérides Kepler válidas de um SV (uma por época)."""
        sub = nav.sel(sv=sv)
        if "sqrtA" not in sub or "Toe" not in sub:
            return []
        try:
            constellation = constellation_from_sv(sv)
        except ValueError:
            return []
        week_field = _WEEK_FIELD.get(constellation)
        if week_field is None or week_field not in sub:
            return []

        valid = np.flatnonzero(np.asarray(sub["sqrtA"].notnull()))
        if valid.size == 0:
            return []

        series: list[BroadcastEphemeris] = []
        canonical = _canonical_sv(sv)
        for idx_raw in valid:
            idx = int(idx_raw)
            week = float(sub[week_field].values[idx])
            if np.isnan(week):
                continue
            toe_sec = float(sub["Toe"].values[idx])
            if week_field == "BDTWeek":
                toe = _bds_week_to_datetime(week, toe_sec)
            else:
                toe = _gps_week_to_datetime(week, toe_sec)
            toc = _to_datetime(sub["time"].values[idx])

            def f(name: str, default: float = 0.0, *, _sub=sub, _idx=idx) -> float:
                if name not in _sub:
                    return default
                val = _sub[name].values[_idx]
                return default if np.isnan(val) else float(val)

            series.append(
                BroadcastEphemeris(
                    sv=canonical,
                    toc=toc,
                    toe=toe,
                    af0=f("SVclockBias"),
                    af1=f("SVclockDrift"),
                    af2=f("SVclockDriftRate"),
                    sqrt_a=f("sqrtA"),
                    e=f("Eccentricity"),
                    i0=f("Io"),
                    omega0=f("Omega0"),
                    omega=f("omega"),
                    m0=f("M0"),
                    delta_n=f("DeltaN"),
                    idot=f("IDOT"),
                    omega_dot=f("OmegaDot"),
                    crc=f("Crc"),
                    crs=f("Crs"),
                    cuc=f("Cuc"),
                    cus=f("Cus"),
                    cic=f("Cic"),
                    cis=f("Cis"),
                    health=int(f("health", 0)),
                    ura=f("SVacc", 0.0),
                )
            )
        return series

    def _sv_to_kepler(
        self,
        nav: xr.Dataset,
        sv: str,
    ) -> BroadcastEphemeris | None:
        """Efeméride Kepler mais recente (última época válida) do SV."""
        series = self._sv_kepler_series(nav, sv)
        return series[-1] if series else None

    def to_ephemeris_series(
        self,
        nav: xr.Dataset,
        sv: str,
    ) -> list[BroadcastEphemeris]:
        """Série completa de efemérides Kepler de um SV (por época).

        Útil para selecionar a efeméride mais próxima de uma janela de
        observação (em vez de sempre a última do arquivo).
        """
        return self._sv_kepler_series(nav, sv)

    def _sv_glonass_series(
        self,
        nav: xr.Dataset,
        sv: str,
    ) -> list[GlonassEphemeris]:
        """Todas as efemérides GLONASS válidas de um SV (uma por época)."""
        if "X" not in nav:
            return []
        sub = nav.sel(sv=sv)
        valid = np.flatnonzero(np.asarray(sub["X"].notnull()))
        if valid.size == 0:
            return []

        canonical = _canonical_sv(sv)
        series: list[GlonassEphemeris] = []
        for idx_raw in valid:
            idx = int(idx_raw)

            def f(name: str, *, _sub=sub, _idx=idx) -> float:
                return float(_sub[name].values[_idx])

            tb = _to_datetime(sub["time"].values[idx])
            series.append(
                GlonassEphemeris(
                    sv=canonical,
                    tb=tb,
                    x=f("X"),
                    y=f("Y"),
                    z=f("Z"),
                    vx=f("dX"),
                    vy=f("dY"),
                    vz=f("dZ"),
                    ax=f("dX2"),
                    ay=f("dY2"),
                    az=f("dZ2"),
                    freq_num=int(f("FreqNum")),
                )
            )
        return series

    def to_glonass_ephemeris_series(
        self,
        nav: xr.Dataset,
        sv: str,
    ) -> list[GlonassEphemeris]:
        """Série completa de efemérides GLONASS de um SV (por época)."""
        return self._sv_glonass_series(nav, sv)

    def to_glonass_ephemerides(self, nav: xr.Dataset) -> list[GlonassEphemeris]:
        """Converte NAV GLONASS em efemérides de vetor de estado.

        Para cada SV mantém a efeméride mais recente (última época válida).
        Use :meth:`to_glonass_ephemeris_series` para obter todas as épocas.
        """
        if "sv" not in nav.coords:
            return []
        ephs: list[GlonassEphemeris] = []
        for sv_raw in nav["sv"].values:
            sv = str(sv_raw)
            if not sv.startswith(("R", "r")):
                continue
            series = self._sv_glonass_series(nav, sv)
            if series:
                ephs.append(series[-1])
        return ephs

    # ------------------------------------------------------------------
    # Observação → C/N0
    # ------------------------------------------------------------------

    def to_signal_samples(
        self,
        obs: xr.Dataset,
        signal: str = "S1C",
    ) -> list[SignalSample]:
        """Extrai amostras C/N0 (dB-Hz) de um dataset de observação.

        Se a variável ``signal`` não existir, tenta detectar automaticamente
        qualquer variável ``S*`` (SNR/C/N0) disponível.

        Parameters
        ----------
        obs : xr.Dataset
            Dataset RINEX de observação (georinex).
        signal : str
            Nome da variável de intensidade (ex.: ``S1C``, ``S2W``).
        """
        var = signal if signal in obs else _autodetect_signal_var(obs)
        if var is None or "time" not in obs.coords:
            return []
        cn0 = obs[var]
        times = cn0["time"].values
        samples: list[SignalSample] = []
        for sv_raw in cn0["sv"].values:
            sv = str(sv_raw)
            values = cn0.sel(sv=sv).values
            for t, val in zip(times, values, strict=False):
                if np.isnan(val):
                    continue
                samples.append(SignalSample(time=_to_datetime(t), sv=sv, cn0_dbhz=float(val)))
        return samples

    def mean_cn0_per_sv(
        self,
        obs: xr.Dataset,
        signal: str = "S1C",  # noqa: ARG002 — o resumo usa o melhor sinal por SV
    ) -> list[tuple[str, float]]:
        """C/N0 médio por satélite (vetorizado) — caminho rápido para resumos.

        Para cada satélite considera o **melhor sinal** (maior média de C/N0
        entre todas as variáveis ``S*`` disponíveis). Isso cobre todas as
        constelações mesmo quando o sinal pedido (ex.: ``S1C``, ausente em
        Beidou, que usa ``S2I``) não é o observado.

        Returns
        -------
        list[(sv, cn0_mean_dbhz)]
            Pares (identificador do satélite, média de C/N0 em dB-Hz).
        """
        if "time" not in obs.coords:
            return []
        s_vars = [v for v in obs.data_vars if v.startswith("S") and "time" in obs[v].dims]
        if not s_vars:
            return []
        # Média por sinal ao longo do tempo → array sobre sv.
        means: list[np.ndarray] = []
        for var in s_vars:
            means.append(np.asarray(obs[var].mean(dim="time", skipna=True).values))
        best = np.nanmax(np.stack(means), axis=0)  # melhor sinal por satélite

        out: list[tuple[str, float]] = []
        for sv_raw, value in zip(obs["sv"].values, best, strict=False):
            if np.isnan(value):
                continue
            out.append((str(sv_raw), float(value)))
        return out


def _autodetect_signal_var(obs: xr.Dataset) -> str | None:
    """Retorna a primeira variável ``S*`` do dataset, se houver."""
    for var in obs.data_vars:
        if var.startswith("S") and obs[var].dims and "time" in obs[var].dims:
            return var
    return None
