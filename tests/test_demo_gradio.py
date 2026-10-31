"""UI-specific contracts. Install requirements-ui.txt to run these checks."""
import importlib.util
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('gradio'), 'Optional Gradio UI dependency is not installed')
class GradioTests(unittest.TestCase):
    def setUp(self):
        from demo_ui.app import build_app
        from demo_ui.controller import Controller
        self.temp=tempfile.TemporaryDirectory()
        self.controller=Controller(output_root=self.temp.name,replay_only=True)
        self.app=build_app(self.controller)

    def tearDown(self):
        self.controller.close_all()
        self.temp.cleanup()

    def test_client_endpoint_names_are_strings_and_timer_is_private(self):
        # Gradio 6 client rejects boolean api_name values when seeking endpoints.
        dependencies=self.app.config['dependencies']
        self.assertTrue(all(d.get('api_name') is None or isinstance(d['api_name'],str) for d in dependencies))
        names={d['api_name'] for d in dependencies}
        self.assertTrue({'load_model','create_episode','next_step','seek_live','infer','replay_load','replay_seek'}<=names)
        timers=[d for d in dependencies if any(event[1]=='tick' for event in d['targets'])]
        self.assertEqual(len(timers),2)
        self.assertTrue(all(d['api_visibility']=='private' for d in timers))

    def test_checkpoint_switch_clears_previous_live_and_inference_results(self):
        callback=next(f for f in self.app.fns.values() if f.fn and f.fn.__name__=='on_load_model')
        meta=dict(checkpoint_name='unit-ui-format-only',state_dim=23,image_shape=[64,64,3],
                  control_source='observed',checkpoint_sha256='unit-fixture')
        with patch.object(self.controller,'load',return_value=meta):
            result=callback.fn('unit-session','unused','','cpu')
        self.assertEqual(len(result),len(callback.outputs))
        self.assertIsNone(result[2]);self.assertIsNone(result[3])
        self.assertIsNone(result[12])
        self.assertEqual(result[-1],[])
        self.assertIn('Khởi tạo episode mới',result[10])


if __name__=='__main__':unittest.main()
