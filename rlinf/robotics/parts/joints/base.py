# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interfaces for joint groups that are not arms."""

from rlinf.robotics.parts.base import ControllablePart


class JointGroup(ControllablePart):
    """A named set of joints commanded by absolute position.

    A humanoid head, torso, or waist is a joint group: it moves in joint space
    like an arm, but it carries no end effector and is not a place a policy
    reaches to. Keeping it out of :class:`Arm` leaves ``parts_of_type(Arm)``
    answering with the arms a task can actually reach with.

    Drivers declare their joint names, travel, and units through the standard
    part interface.
    """
