"""Regression checks for RASAero attachment planes and fin/boattail placement."""

import unittest
import xml.etree.ElementTree as ET

from cdx1_summary_png import (
    _additional_input_cards, _fin_root_le, _parse_fin, _parse_part,
    _part_polygon, _resolve_geometry,
)


def geometry(xml):
    parts = [_parse_part(node, i) for i, node in enumerate(ET.fromstring(xml), 1)]
    _resolve_geometry(parts)
    return parts


class ProfileGeometryTests(unittest.TestCase):
    def test_tip_and_fin_radii_distinguish_zero_from_missing(self):
        nose = _parse_part(ET.fromstring('<NoseCone><BluntRadius>0.125</BluntRadius></NoseCone>'), 1)
        self.assertEqual(nose['tip_radius'], 0.125)
        self.assertEqual(_parse_fin(ET.fromstring('<Fin><LERadius>0</LERadius></Fin>'))['le_radius'], 0)
        self.assertIsNone(_parse_fin(ET.fromstring('<Fin/>'))['le_radius'])

    def test_additional_input_cards_only_show_requested_fields(self):
        root = ET.fromstring('''<RASAeroDocument><RocketDesign>
          <BodyTube><Length>20</Length><RailGuideHeight>0.6</RailGuideHeight>
            <RailGuideDiameter>0.7</RailGuideDiameter><LaunchShoeArea>0</LaunchShoeArea>
            <BoattailLength>0</BoattailLength><Location>10</Location>
            <Protuberance><StreamlinedWithBaseDrag>0.65</StreamlinedWithBaseDrag></Protuberance>
            <Fin><LERadius>0.01</LERadius><FX1>0.75</FX1><Location>8</Location></Fin>
            <Fin><FX1>0.5</FX1></Fin></BodyTube>
          <ModifiedBarrowman>True</ModifiedBarrowman>
          <Turbulence>False</Turbulence><FutureSetting>0</FutureSetting>
        </RocketDesign><MachAlt><Point><Mach>8</Mach><Altitude>120000</Altitude></Point></MachAlt>
        <SimulationList><Simulation><ExtraDelay>2.5</ExtraDelay></Simulation></SimulationList>
        </RASAeroDocument>''')
        cards = _additional_input_cards(root)
        fields = [field for _, rows in cards for field in rows]
        self.assertEqual(fields, [
            ['ModifiedBarrowman', 'True'], ['Turbulence', 'False'],
            ['RailGuideDiameter (in)', '0.7'], ['LaunchShoeArea (in²)', '0'],
            ['StreamlinedWithBaseDrag', '0.65'],
        ])
        self.assertTrue(all(rows for _, rows in cards))

    def test_fin_can_ends_where_expanding_booster_begins(self):
        tube, sleeve, booster = geometry("""<Design>
          <BodyTube><Location>15</Location><Length>53</Length><Diameter>2.91</Diameter></BodyTube>
          <FinCan><Location>68</Location><Length>8</Length><ShoulderLength>1</ShoulderLength>
            <Diameter>3.035</Diameter><InsideDiameter>2.91</InsideDiameter>
            <Fin><Location>8</Location><Chord>8</Chord></Fin></FinCan>
          <Booster><Location>68</Location><Length>58.5</Length><ShoulderLength>2</ShoulderLength>
            <Diameter>4.25</Diameter><InsideDiameter>3.035</InsideDiameter>
            <BoattailLength>2</BoattailLength><BoattailRearDiameter>3.828</BoattailRearDiameter>
            <Fin><Location>10</Location><Chord>10</Chord></Fin></Booster>
        </Design>""")
        self.assertEqual(sleeve['profile_start'], 59)
        self.assertEqual(_fin_root_le(sleeve, sleeve['fins'][0]), 60)
        self.assertEqual(sleeve['profile_end'], tube['body_end'])
        self.assertEqual(sleeve['profile_end'], booster['profile_start'])
        self.assertEqual(_part_polygon(booster)[:2], [(68, 3.035 / 2), (70, 4.25 / 2)])
        root_aft = _fin_root_le(booster, booster['fins'][0]) + 10
        self.assertEqual(root_aft, 128.5)
        self.assertIn((root_aft, 4.25 / 2), _part_polygon(booster))
        self.assertIn((130.5, 3.828 / 2), _part_polygon(booster))
        self.assertEqual(booster['profile_end'], 130.5)

    def test_setback_fin_keeps_gap_before_boattail(self):
        part, = geometry("""<Design><Booster>
          <Location>20</Location><Length>40</Length><Diameter>4</Diameter>
          <BoattailLength>3</BoattailLength><BoattailRearDiameter>3</BoattailRearDiameter>
          <Fin><Location>12</Location><Chord>10</Chord></Fin>
        </Booster></Design>""")
        self.assertEqual(_fin_root_le(part, part['fins'][0]) + 10, 58)
        self.assertIn((60, 2), _part_polygon(part))
        self.assertEqual(part['profile_end'], 63)

    def test_standalone_transition_matches_preceding_tube(self):
        _, transition = geometry("""<Design>
          <BodyTube><Location>10</Location><Length>20</Length><Diameter>3</Diameter></BodyTube>
          <Transition><Location>30</Location><Length>2</Length>
            <Diameter>4</Diameter><RearDiameter>4</RearDiameter></Transition>
        </Design>""")
        self.assertEqual(_part_polygon(transition), [(30, 1.5), (32, 2), (32, -2), (30, -1.5)])


if __name__ == '__main__':
    unittest.main()
