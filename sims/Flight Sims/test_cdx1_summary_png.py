"""Regression checks for RASAero attachment planes and fin/boattail placement."""

import unittest
import xml.etree.ElementTree as ET

from cdx1_summary_png import _fin_root_le, _parse_part, _part_polygon, _resolve_geometry


def geometry(xml):
    parts = [_parse_part(node, i) for i, node in enumerate(ET.fromstring(xml), 1)]
    _resolve_geometry(parts)
    return parts


class ProfileGeometryTests(unittest.TestCase):
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
