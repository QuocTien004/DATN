"""Portable replay contracts, error handling and source-based comparisons."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from demo_ui.recording import EpisodeWriter, EpisodeReader, export_episode, import_episode
from demo_ui.analytics import compare_evaluations, read_training_log
from demo_ui.controller import Controller
from demo_ui.presentation import metrics_html, prediction_rows, sensor_rows


class ReplayTests(unittest.TestCase):
    def test_signal_panel_compares_prior_with_result_of_same_action(self):
        row=dict(route_completion=.42,action_executed=[-.25,.75],reward=-2.5,
                 continue_target=1.,episode_return=10.,
                 prior={'reward':-1.25,'continue_probability':.875},
                 posterior_next={'reward':999.,'continue_probability':.123})
        view=metrics_html(row)
        for value in ['42.0%','-0.250','0.750','-1.250','-2.500','0.875','1.000']:
            self.assertIn(value,view)
        self.assertNotIn('999.000',view)
        self.assertNotIn('0.123',view)

    def test_empty_signal_panel_does_not_invent_zero_results(self):
        view=metrics_html()
        self.assertIn('Chưa có episode',view)
        self.assertIn('—',view)
        self.assertNotIn('<progress',view)
        self.assertNotIn('0.000',view)

    def make_recording(self, root):
        # Unit fixture only, never offered as model/demo evidence.
        writer=EpisodeWriter(root,dict(map_seed=7,rng_seed=0,safety_shield=False))
        rgb=np.arange(12*16*3,dtype=np.uint8).reshape(12,16,3)
        row=dict(state=[0.]*23,action_actor=[.1,.2],action_executed=[.1,-1.],
                 reward=-2.,continue_target=0.,terminated=True,truncated=False,
                 prior={'reward':-1.,'continue_probability':.3},
                 posterior_next={'reward':-1.8,'continue_probability':.1})
        writer.append(row,rgb,rgb)
        writer.close(complete=True,reason='terminated')
        return writer.folder,rgb

    def test_lossless_recording_export_import_and_lazy_seek(self):
        with tempfile.TemporaryDirectory() as td:
            folder,rgb=self.make_recording(td)
            reader=EpisodeReader(folder)
            self.assertEqual(len(reader),1)
            row,actual,top=reader.frame(0)
            np.testing.assert_array_equal(actual,rgb);np.testing.assert_array_equal(top,rgb)
            self.assertEqual(row['action_executed'],[.1,-1.])
            imported=import_episode(export_episode(folder),Path(td)/'import')
            np.testing.assert_array_equal(imported.frame(0)[1],rgb)
            self.assertTrue(imported.metadata['complete'])
            with self.assertRaises(IndexError):reader.frame(1)

    def test_checksum_rejects_changed_sensor_record_or_frame(self):
        with tempfile.TemporaryDirectory() as td:
            folder,rgb=self.make_recording(td)
            Image.fromarray(np.zeros_like(rgb)).save(folder/'frames/000000_rgb.png')
            with self.assertRaisesRegex(ValueError,'Checksum'):EpisodeReader(folder).frame(0)
            with (folder/'steps.jsonl').open('a') as file:file.write('{}\n')
            with self.assertRaisesRegex(ValueError,'Checksum'):EpisodeReader(folder)

    def test_recover_incomplete_recording_and_missing_topdown(self):
        with tempfile.TemporaryDirectory() as td:
            writer=EpisodeWriter(td,{})
            writer.append(dict(state=[0.]*19),np.zeros((12,12,3),dtype=np.uint8))
            with (writer.folder/'steps.jsonl').open('a') as file:file.write('{partial')
            reader=EpisodeReader(writer.folder)
            self.assertEqual(len(reader),1)
            self.assertIsNone(reader.frame(0)[2])
            writer.close()

    def test_zip_cannot_escape_output_directory(self):
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'bad.zip'
            with zipfile.ZipFile(file,'w') as archive:archive.writestr('../escape.json','{}')
            with self.assertRaises(ValueError):import_episode(file,Path(td)/'out')
            self.assertFalse((Path(td)/'escape.json').exists())

    def test_replay_mode_isolated_between_browser_sessions_and_no_worker(self):
        with tempfile.TemporaryDirectory() as td:
            folder,_=self.make_recording(td)
            ctx=Controller(output_root=Path(td)/'runs',replay_only=True)
            ctx.load_replay('one',path=folder)
            self.assertIsNotNone(ctx.get('one').replay)
            self.assertIsNone(ctx.get('two').replay)
            self.assertIsNone(ctx.get('one').worker)
            with self.assertRaisesRegex(ValueError,'Replay CPU'):ctx.load('two','any',None,'cpu')
            ctx.close_all()

    def test_replay_imports_without_torch_or_metadrive(self):
        code='import sys; import demo_ui.recording,demo_ui.controller,demo_ui.analytics; assert "torch" not in sys.modules; assert "metadrive" not in sys.modules'
        subprocess.run([sys.executable,'-c',code],check=True)

    def test_live_history_seek_keeps_full_length_and_does_not_move_simulator(self):
        with tempfile.TemporaryDirectory() as td:
            writer=EpisodeWriter(td,{})
            for _ in range(3):writer.append(dict(state=[0.]*23),np.zeros((12,12,3),dtype=np.uint8))
            writer.close()
            ctx=Controller(output_root=Path(td)/'out')
            session=ctx.get('live');session.folder=str(writer.folder)
            session.frame={'record':{'step':2}}
            self.assertEqual(ctx.seek_live('live',0)['length'],3)
            self.assertEqual(ctx.seek_live('live',2)['record']['step'],2)
            self.assertEqual(session.frame['record']['step'],2)
            session.running=True
            with self.assertRaisesRegex(ValueError,'Tạm dừng'):ctx.seek_live('live',1)
            ctx.close_all()

    def test_prediction_columns_keep_prior_posterior_and_terminal_timing(self):
        rows=prediction_rows(dict(reward=-2.,continue_target=0.,
            prior={'reward':-1.,'continue_probability':.3},
            posterior_next={'reward':-1.8,'continue_probability':.1}))
        self.assertEqual(rows[0][1:],[-1.,-1.8,-2.])
        self.assertEqual(rows[1][1:],[.3,.1,0.])
        self.assertIn('steering',sensor_rows([0.]*23)[5][1])
        self.assertIn('ga/phanh',sensor_rows([0.]*23)[6][1])

    def test_eval_recomputes_episodes_and_flags_unpaired_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'a.json';b=Path(td)/'b.json'
            def write(file,seed,route):
                file.write_text(json.dumps({'horizon':1000,'aggregate':{'mean_route_completion':999},
                    'episodes':[dict(seed=seed,route_completion=route,success=0.,collision=1.,out_of_road=0.)]}))
            write(a,7,.4);write(b,8,.6)
            rows,note=compare_evaluations(a,b)
            self.assertEqual(rows[0][2],40.)
            self.assertIn('khác seed/horizon',note)
            self.assertIn('thiếu checkpoint hash',note)

    def test_eval_without_horizon_is_not_reported_as_matching_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'evaluation.json'
            file.write_text(json.dumps({'episodes':[dict(seed=7,route_completion=.4,
                success=0.,collision=0.,out_of_road=0.)]}))
            _,note=compare_evaluations(file,file)
            self.assertIn('thiếu hoặc khác seed/horizon',note)
            self.assertNotIn('horizon khớp',note)

    def test_training_axis_preserves_environment_step_and_update_keys(self):
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'metrics.jsonl'
            file.write_text('{"env_steps":30000,"updates":750,"loss":2}\n')
            rows,keys=read_training_log(file)
            self.assertEqual(rows[0]['env_steps'],30000)
            self.assertEqual(rows[0]['updates'],750)
            self.assertIn('env_steps',keys)
            with self.assertRaises(ValueError):read_training_log(file,limit=0)


if __name__=='__main__':unittest.main()
