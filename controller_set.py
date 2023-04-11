from manual_controller import ManualController

MANUAL_CONTROL = True

# These represent incremental movements of the thumbnail that guides the robots.
# MAX_FORWARD_SHIFT is in units of pixels of the output image.  
# MAX_ANGULAR_SHIFT is in radians.
MAX_FORWARD_SHIFT = -10
MAX_ANGULAR_SHIFT = -0.1

class ControllerSet:
    def __init__(self):
        if MANUAL_CONTROL:
            self.manual_controller = ManualController()
    
    def get_movements(self, decoded_tags):

        movement_dict = {}

        for tag in decoded_tags:
            if not tag['is_valid']:
                continue

            assert 'tag_id' in tag

            # TBD: Filter other tags corresponding to anything but active robots.

            #movement_dict[tag['tag_id']] = MAX_FORWARD_SHIFT, MAX_ANGULAR_SHIFT
            movement_dict[tag['tag_id']] = self.manual_controller.get_forward_angular_tuple()

        return movement_dict