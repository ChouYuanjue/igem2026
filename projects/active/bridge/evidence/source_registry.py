from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    title: str
    observation_families: tuple[str, ...]
    identifier_kinds: tuple[str, ...]
    access_mode: str
    auth_mode: str
    cache_ttl_seconds: int
    rate_limit_hz: float | None
    documentation_url: str
    license_note: str

    def to_dict(self) -> dict:
        return asdict(self)


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        'uniprot','UniProtKB',
        ('protein_annotation','catalytic_annotation','cofactor','sequence_feature','cross_reference'),
        ('uniprot','protein_name','ec'),
        'rest','public',7*86400,None,
        'https://www.uniprot.org/help/programmatic_access',
        'public programmatic access; preserve returned annotation evidence',
    ),
    SourceSpec(
        'rhea','Rhea',
        ('reaction_standard','reaction_participant','reaction_direction','reaction_cross_reference'),
        ('rhea','chebi','ec','text'),
        'rest+sparql','public',14*86400,None,
        'https://www.rhea-db.org/help/api',
        'public reaction resource; preserve directed/master reaction semantics',
    ),
    SourceSpec(
        'europe_pmc','Europe PMC',
        ('literature','full_text','assay_context','citation'),
        ('pmid','pmcid','doi','text','uniprot','rhea'),
        'rest','public',7*86400,None,
        'https://europepmc.org/RestfulWebService',
        'metadata broadly available; full-text availability depends on article rights',
    ),
    SourceSpec(
        'sabio_rk','SABIO-RK',
        ('kinetics','assay_context','rate_law','reaction_participant'),
        ('uniprot','ec','rhea','organism','compound','text'),
        'rest','public',7*86400,None,
        'https://www.sabio.h-its.org/layouts/content/docuRESTfulWeb/index.gsp',
        'curated kinetics database; compound structure identifiers require independent validation',
    ),
    SourceSpec(
        'brenda','BRENDA',
        ('kinetics','assay_context','cofactor','inhibitor','stability','enzyme_annotation'),
        ('ec','organism','pmid','compound'),
        'soap+bulk_json','registered',30*86400,1.0,
        'https://www.brenda-enzymes.org/soap.php',
        'CC BY 4.0; SOAP requires registered credentials; bulk download requires license acceptance',
    ),
    SourceSpec(
        'mcsa','M-CSA',
        ('mechanism','catalytic_residue','cofactor','mechanism_reaction','mechanism_homology'),
        ('uniprot','ec','pdb','pmid','mcsa'),
        'rest','public',30*86400,None,
        'https://www.ebi.ac.uk/thornton-srv/m-csa/download/',
        'CC BY 4.0; curated and homology-derived residues must remain distinguishable',
    ),
    SourceSpec(
        'pdbe','PDBe',
        ('experimental_structure','binding_site','ligand','structure_annotation','sifts_mapping'),
        ('pdb','uniprot','ligand'),
        'rest','public',7*86400,None,
        'https://www.ebi.ac.uk/pdbe/pdbe-rest-api',
        'public structural archive/API; use current unified PDBe endpoints',
    ),
    SourceSpec(
        'alphafold_db','AlphaFold DB',
        ('predicted_structure','structure_confidence'),
        ('uniprot',),
        'rest','public',30*86400,None,
        'https://alphafold.ebi.ac.uk/',
        'predicted structure; confidence/provenance must remain distinct from experimental PDB evidence',
    ),
    SourceSpec(
        'chebi','ChEBI',
        ('chemical_identity','chemical_structure','chemical_ontology','chemical_cross_reference'),
        ('chebi','compound_name'),
        'rest','public',30*86400,None,
        'https://www.ebi.ac.uk/chebi/tools',
        'public chemical ontology and entity API',
    ),
    SourceSpec(
        'pubchem','PubChem',
        ('chemical_identity','chemical_structure','chemical_annotation','bioassay'),
        ('pubchem_cid','compound_name','inchikey'),
        'pug_rest+pug_view','public',14*86400,5.0,
        'https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest',
        'public service; respect request limits and preserve third-party annotation provenance',
    ),
)

SOURCE_BY_ID={x.source_id:x for x in SOURCE_SPECS}


def sources_for(
    observation_families: Iterable[str],
    available_identifier_kinds: Iterable[str],
    *,
    include_registered: bool = True,
) -> tuple[SourceSpec, ...]:
    families={str(x) for x in observation_families if str(x)}
    identifiers={str(x) for x in available_identifier_kinds if str(x)}
    out=[]
    for spec in SOURCE_SPECS:
        if not families.intersection(spec.observation_families):
            continue
        if not identifiers.intersection(spec.identifier_kinds):
            continue
        if spec.auth_mode == 'registered' and not include_registered:
            continue
        out.append(spec)
    return tuple(out)


def capability_matrix() -> dict[str, list[str]]:
    matrix: dict[str,list[str]]={}
    for spec in SOURCE_SPECS:
        for family in spec.observation_families:
            matrix.setdefault(family,[]).append(spec.source_id)
    return {key:sorted(value) for key,value in sorted(matrix.items())}
