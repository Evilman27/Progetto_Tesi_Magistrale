from setuptools import setup
package_name = 'music_emotion_node'
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': ['ros2_bridge = music_emotion_node.live_emotion_pub:main',
                                      'ActionExecutorQi = music_emotion_node.action_executor_node:main',
                                      'ActionCodegenExecutorQi = music_emotion_node.action_codegen_executor_node:main',]
    },
)

