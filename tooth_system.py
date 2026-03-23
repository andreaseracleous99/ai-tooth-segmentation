"""
Tooth Classification System using FDI Numbering System

FDI System (32 permanent teeth):
- 11-18: Maxillary right (upper right)
- 21-28: Maxillary left (upper left)  
- 31-38: Mandibular left (lower left)
- 41-48: Mandibular right (lower right)

Each tooth position has a name:
11: Central incisor (max right), 12: Lateral incisor (max right), 13: Canine (max right)
14: First premolar (max right), 15: Second premolar (max right)
16: First molar (max right), 17: Second molar (max right), 18: Third molar (max right)

Similar pattern for 21-28 (maxillary left), 31-38 (mandibular left), 41-48 (mandibular right)
"""

from enum import Enum
from typing import List, Dict, Tuple


class ToothGroup(Enum):
    """Classification of teeth into functional groups"""
    INCISOR = "incisor"
    CANINE = "canine"
    PREMOLAR = "premolar"
    MOLAR = "molar"


class Arch(Enum):
    """Dental arch classification"""
    MAXILLARY_RIGHT = "maxillary_right"
    MAXILLARY_LEFT = "maxillary_left"
    MANDIBULAR_LEFT = "mandibular_left"
    MANDIBULAR_RIGHT = "mandibular_right"


class ToothPosition(Enum):
    """Individual tooth position in arch"""
    CENTRAL_INCISOR = "central_incisor"
    LATERAL_INCISOR = "lateral_incisor"
    CANINE = "canine"
    FIRST_PREMOLAR = "first_premolar"
    SECOND_PREMOLAR = "second_premolar"
    FIRST_MOLAR = "first_molar"
    SECOND_MOLAR = "second_molar"
    THIRD_MOLAR = "third_molar"


# FDI Tooth numbering system
FDI_TEETH = {
    # Maxillary right (11-18)
    11: {
        "fdi": 11,
        "name": "Central Incisor",
        "arch": Arch.MAXILLARY_RIGHT,
        "position": ToothPosition.CENTRAL_INCISOR,
        "group": ToothGroup.INCISOR,
        "side": "right",
    },
    12: {
        "fdi": 12,
        "name": "Lateral Incisor",
        "arch": Arch.MAXILLARY_RIGHT,
        "position": ToothPosition.LATERAL_INCISOR,
        "group": ToothGroup.INCISOR,
        "side": "right",
    },
    13: {
        "fdi": 13,
        "name": "Canine",
        "arch": Arch.MAXILLARY_RIGHT,
        "position": ToothPosition.CANINE,
        "group": ToothGroup.CANINE,
        "side": "right",
    },
    14: {
        "fdi": 14,
        "name": "First Premolar",
        "arch": Arch.MAXILLARY_RIGHT,
        "position": ToothPosition.FIRST_PREMOLAR,
        "group": ToothGroup.PREMOLAR,
        "side": "right",
    },
    15: {
        "fdi": 15,
        "name": "Second Premolar",
        "arch": Arch.MAXILLARY_RIGHT,
        "position": ToothPosition.SECOND_PREMOLAR,
        "group": ToothGroup.PREMOLAR,
        "side": "right",
    },
    16: {
        "fdi": 16,
        "name": "First Molar",
        "arch": Arch.MAXILLARY_RIGHT,
        "position": ToothPosition.FIRST_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "right",
    },
    17: {
        "fdi": 17,
        "name": "Second Molar",
        "arch": Arch.MAXILLARY_RIGHT,
        "position": ToothPosition.SECOND_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "right",
    },
    18: {
        "fdi": 18,
        "name": "Third Molar",
        "arch": Arch.MAXILLARY_RIGHT,
        "position": ToothPosition.THIRD_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "right",
    },
    # Maxillary left (21-28)
    21: {
        "fdi": 21,
        "name": "Central Incisor",
        "arch": Arch.MAXILLARY_LEFT,
        "position": ToothPosition.CENTRAL_INCISOR,
        "group": ToothGroup.INCISOR,
        "side": "left",
    },
    22: {
        "fdi": 22,
        "name": "Lateral Incisor",
        "arch": Arch.MAXILLARY_LEFT,
        "position": ToothPosition.LATERAL_INCISOR,
        "group": ToothGroup.INCISOR,
        "side": "left",
    },
    23: {
        "fdi": 23,
        "name": "Canine",
        "arch": Arch.MAXILLARY_LEFT,
        "position": ToothPosition.CANINE,
        "group": ToothGroup.CANINE,
        "side": "left",
    },
    24: {
        "fdi": 24,
        "name": "First Premolar",
        "arch": Arch.MAXILLARY_LEFT,
        "position": ToothPosition.FIRST_PREMOLAR,
        "group": ToothGroup.PREMOLAR,
        "side": "left",
    },
    25: {
        "fdi": 25,
        "name": "Second Premolar",
        "arch": Arch.MAXILLARY_LEFT,
        "position": ToothPosition.SECOND_PREMOLAR,
        "group": ToothGroup.PREMOLAR,
        "side": "left",
    },
    26: {
        "fdi": 26,
        "name": "First Molar",
        "arch": Arch.MAXILLARY_LEFT,
        "position": ToothPosition.FIRST_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "left",
    },
    27: {
        "fdi": 27,
        "name": "Second Molar",
        "arch": Arch.MAXILLARY_LEFT,
        "position": ToothPosition.SECOND_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "left",
    },
    28: {
        "fdi": 28,
        "name": "Third Molar",
        "arch": Arch.MAXILLARY_LEFT,
        "position": ToothPosition.THIRD_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "left",
    },
    # Mandibular left (31-38)
    31: {
        "fdi": 31,
        "name": "Central Incisor",
        "arch": Arch.MANDIBULAR_LEFT,
        "position": ToothPosition.CENTRAL_INCISOR,
        "group": ToothGroup.INCISOR,
        "side": "left",
    },
    32: {
        "fdi": 32,
        "name": "Lateral Incisor",
        "arch": Arch.MANDIBULAR_LEFT,
        "position": ToothPosition.LATERAL_INCISOR,
        "group": ToothGroup.INCISOR,
        "side": "left",
    },
    33: {
        "fdi": 33,
        "name": "Canine",
        "arch": Arch.MANDIBULAR_LEFT,
        "position": ToothPosition.CANINE,
        "group": ToothGroup.CANINE,
        "side": "left",
    },
    34: {
        "fdi": 34,
        "name": "First Premolar",
        "arch": Arch.MANDIBULAR_LEFT,
        "position": ToothPosition.FIRST_PREMOLAR,
        "group": ToothGroup.PREMOLAR,
        "side": "left",
    },
    35: {
        "fdi": 35,
        "name": "Second Premolar",
        "arch": Arch.MANDIBULAR_LEFT,
        "position": ToothPosition.SECOND_PREMOLAR,
        "group": ToothGroup.PREMOLAR,
        "side": "left",
    },
    36: {
        "fdi": 36,
        "name": "First Molar",
        "arch": Arch.MANDIBULAR_LEFT,
        "position": ToothPosition.FIRST_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "left",
    },
    37: {
        "fdi": 37,
        "name": "Second Molar",
        "arch": Arch.MANDIBULAR_LEFT,
        "position": ToothPosition.SECOND_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "left",
    },
    38: {
        "fdi": 38,
        "name": "Third Molar",
        "arch": Arch.MANDIBULAR_LEFT,
        "position": ToothPosition.THIRD_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "left",
    },
    # Mandibular right (41-48)
    41: {
        "fdi": 41,
        "name": "Central Incisor",
        "arch": Arch.MANDIBULAR_RIGHT,
        "position": ToothPosition.CENTRAL_INCISOR,
        "group": ToothGroup.INCISOR,
        "side": "right",
    },
    42: {
        "fdi": 42,
        "name": "Lateral Incisor",
        "arch": Arch.MANDIBULAR_RIGHT,
        "position": ToothPosition.LATERAL_INCISOR,
        "group": ToothGroup.INCISOR,
        "side": "right",
    },
    43: {
        "fdi": 43,
        "name": "Canine",
        "arch": Arch.MANDIBULAR_RIGHT,
        "position": ToothPosition.CANINE,
        "group": ToothGroup.CANINE,
        "side": "right",
    },
    44: {
        "fdi": 44,
        "name": "First Premolar",
        "arch": Arch.MANDIBULAR_RIGHT,
        "position": ToothPosition.FIRST_PREMOLAR,
        "group": ToothGroup.PREMOLAR,
        "side": "right",
    },
    45: {
        "fdi": 45,
        "name": "Second Premolar",
        "arch": Arch.MANDIBULAR_RIGHT,
        "position": ToothPosition.SECOND_PREMOLAR,
        "group": ToothGroup.PREMOLAR,
        "side": "right",
    },
    46: {
        "fdi": 46,
        "name": "First Molar",
        "arch": Arch.MANDIBULAR_RIGHT,
        "position": ToothPosition.FIRST_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "right",
    },
    47: {
        "fdi": 47,
        "name": "Second Molar",
        "arch": Arch.MANDIBULAR_RIGHT,
        "position": ToothPosition.SECOND_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "right",
    },
    48: {
        "fdi": 48,
        "name": "Third Molar",
        "arch": Arch.MANDIBULAR_RIGHT,
        "position": ToothPosition.THIRD_MOLAR,
        "group": ToothGroup.MOLAR,
        "side": "right",
    },
}


def get_tooth_info(fdi_number: int) -> Dict:
    """Get information about a specific tooth by FDI number"""
    if fdi_number not in FDI_TEETH:
        raise ValueError(f"Invalid FDI tooth number: {fdi_number}. Must be 11-18, 21-28, 31-38, or 41-48")
    return FDI_TEETH[fdi_number]


def get_teeth_by_group(group: ToothGroup) -> List[int]:
    """Get all FDI numbers for a specific tooth group"""
    return [fdi for fdi, info in FDI_TEETH.items() if info["group"] == group]


def get_teeth_by_arch(arch: Arch) -> List[int]:
    """Get all FDI numbers for a specific arch"""
    return [fdi for fdi, info in FDI_TEETH.items() if info["arch"] == arch]


def get_all_fdi_numbers() -> List[int]:
    """Get all valid FDI tooth numbers"""
    return sorted(list(FDI_TEETH.keys()))


def get_tooth_label(fdi_number: int) -> str:
    """Get a human-readable label for a tooth"""
    info = get_tooth_info(fdi_number)
    arch_short = info["arch"].value.split("_")[0][0].upper() + info["arch"].value.split("_")[-1][0].upper()
    return f"{info['name']} ({fdi_number}) - {arch_short}"


# Summary statistics
TOOTH_GROUPS_STATS = {
    ToothGroup.INCISOR: {
        "count": len(get_teeth_by_group(ToothGroup.INCISOR)),
        "fdi_numbers": get_teeth_by_group(ToothGroup.INCISOR),
        "description": "Front teeth for cutting (8 teeth total: 2 per quadrant)"
    },
    ToothGroup.CANINE: {
        "count": len(get_teeth_by_group(ToothGroup.CANINE)),
        "fdi_numbers": get_teeth_by_group(ToothGroup.CANINE),
        "description": "Pointed teeth for tearing (4 teeth total: 1 per quadrant)"
    },
    ToothGroup.PREMOLAR: {
        "count": len(get_teeth_by_group(ToothGroup.PREMOLAR)),
        "fdi_numbers": get_teeth_by_group(ToothGroup.PREMOLAR),
        "description": "Grinding teeth (8 teeth total: 2 per quadrant)"
    },
    ToothGroup.MOLAR: {
        "count": len(get_teeth_by_group(ToothGroup.MOLAR)),
        "fdi_numbers": get_teeth_by_group(ToothGroup.MOLAR),
        "description": "Large grinding teeth (12 teeth total: 3 per quadrant)"
    },
}

if __name__ == "__main__":
    print("=== TOOTH SYSTEM CONFIGURATION ===")
    print(f"Total teeth: {len(FDI_TEETH)}")
    print("\nTooth Groups:")
    for group, stats in TOOTH_GROUPS_STATS.items():
        print(f"  {group.value.upper()}: {stats['count']} teeth")
        print(f"    {stats['description']}")
    
    print("\nAll FDI numbers:", get_all_fdi_numbers())
    print("\nSample tooth info:")
    print(f"  Tooth 11: {get_tooth_info(11)}")
    print(f"  Tooth 46: {get_tooth_info(46)}")
