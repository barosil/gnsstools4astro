"""Builders de URL RINEX por fonte (RBMC/IBGE e CDDIS)."""

from __future__ import annotations

from datetime import datetime, timedelta

from gnsstools4astro.domain.rinex import RinexFileType, RinexRequest, RinexSource

# RBMC/IBGE — raiz do serviço (padrão). Os dados ficam em ``/dados/{year}/{doy}``.
RBMC_BASE = "https://geoftp.ibge.gov.br/informacoes_sobre_posicionamento_geodesico/rbmc"

# RBMC/IBGE — RINEX 3 (padrão para observações): ``dados_RINEX3/{year}/{doy}``.
# Arquivos diários de 15 s em CRINEX (``_MO.crx.gz``), ID de 9 caracteres.
RBMC3_BASE = f"{RBMC_BASE}/dados_RINEX3"

# Amostragem padrão dos arquivos RINEX 3 da RBMC (parte do nome do arquivo).
RBMC3_SAMPLING = "15S"

# CDDIS/NASA — arquivos RINEX 3.03 diários (requer credenciais Earthdata).
CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/data/daily"


def _doy(dt: datetime) -> str:
    """Dia do ano com 3 dígitos."""
    return f"{dt.timetuple().tm_yday:03d}"


def _yy(dt: datetime) -> str:
    """Ano com 2 dígitos."""
    return f"{dt.year % 100:02d}"


def _station9(station: str) -> str:
    """ID RINEX 3 de 9 caracteres (padrão RBMC: ``{STATION}00BRA``)."""
    return f"{station.upper()}00BRA"


def build_rbmc_url(
    request: RinexRequest,
    file_type: RinexFileType,
    dt: datetime,
) -> str:
    """URL RBMC/IBGE por tipo de arquivo.

    * **Observação** → RINEX 3 (padrão): ``dados_RINEX3/{year}/{doy}/`` com
      arquivo diário de 15 s em CRINEX (``{STATION}00BRA_R_..._MO.crx.gz``);
    * **Navegação** → RINEX 2 (legado): zip diário ``dados/{year}/{doy}/``
      contendo os NAV (``.24n``/``.24g``/``.24l``), pois a RBMC não publica
      NAV em RINEX 3.
    """
    doy = _doy(dt)
    if file_type is RinexFileType.OBS:
        station9 = _station9(request.station)
        return (
            f"{RBMC3_BASE}/{dt.year}/{doy}/"
            f"{station9}_R_{dt.year}{doy}0000_01D_{RBMC3_SAMPLING}_MO.crx.gz"
        )
    filename = f"{request.station}{doy}1.zip"
    return f"{RBMC_BASE}/dados/{dt.year}/{doy}/{filename}"


def build_cddis_url(
    request: RinexRequest,
    file_type: RinexFileType,
    dt: datetime,
) -> str:
    """URL CDDIS (RINEX 3.03 diário, 30 s).

    Pattern::

        {base}/daily/{year}/{doy}/{yy}o|{yy}n/
            {SSSS00RIN_R_{YYYYDOY0000_01D_30S_MO|MN.rnx.gz}
    """
    doy = _doy(dt)
    yy = _yy(dt)
    sub = "o" if file_type is RinexFileType.OBS else "n"
    tail = "MO" if file_type is RinexFileType.OBS else "MN"

    station_id = request.station.upper()
    if len(station_id) < 9:
        # RINEX 3 usa IDs de 9 caracteres; completa com zeros (estilo CDDIS).
        station_id = station_id.ljust(9, "0")

    name = f"{station_id}_R_{dt.year}{doy}0000_01D_30S_{tail}.rnx.gz"
    return f"{CDDIS_BASE}/{dt.year}/{doy}/{yy}{sub}/{name}"


def build_cddis_brdc_url(dt: datetime) -> str:
    """URL do NAV RINEX 3 **combinado do IGS** (``BRDC00IGS_R``) no CDDIS.

    É a efeméride de transmissão padrão em RINEX 3 e cobre todas as
    constelações (GPS/GLONASS/GALILEO/BEIDOU/QZSS). Fica no subdiretório
    ``{yy}p`` do CDDIS (NAV RINEX 3; ``{yy}n`` é o NAV RINEX 2). Disponível a
    partir do fim de 2024 (migração do IGS para RINEX 3 diário); para datas
    anteriores o downloader cai para o NAV RINEX 2.

    Pattern::

        {base}/daily/{year}/{doy}/{yy}p/BRDC00IGS_R_{YYYYDOY0000_01D_MN.rnx.gz
    """
    doy = _doy(dt)
    yy = _yy(dt)
    return f"{CDDIS_BASE}/{dt.year}/{doy}/{yy}p/BRDC00IGS_R_{dt.year}{doy}0000_01D_MN.rnx.gz"


def build_url(
    request: RinexRequest,
    file_type: RinexFileType,
    dt: datetime,
) -> str:
    """Delega para o builder da fonte configurada no pedido."""
    if request.source is RinexSource.RBMC:
        return build_rbmc_url(request, file_type, dt)
    if request.source is RinexSource.CDDIS:
        return build_cddis_url(request, file_type, dt)
    msg = f"fonte desconhecida: {request.source}"
    raise ValueError(msg)


def list_urls(
    request: RinexRequest,
    file_type: RinexFileType,
) -> list[tuple[datetime, str]]:
    """Gera as URLs por dia coberto pelo intervalo (um arquivo por dia).

    Returns
    -------
    list[(datetime, str)]
        Pares (meia-noite UTC do dia, url).
    """
    out: list[tuple[datetime, str]] = []
    day = request.start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < request.end:
        out.append((day, build_url(request, file_type, day)))
        day += timedelta(days=1)
    return out


class RinexSourceAdapter:
    """Adapta os builders de URL ao port :class:`RinexSourcePort`.

    A fonte (RBMC/CDDIS) é determinada pelo próprio pedido.
    """

    def build_url(
        self,
        request: RinexRequest,
        file_type: RinexFileType,
        day: datetime,
    ) -> str:
        return build_url(request, file_type, day)
