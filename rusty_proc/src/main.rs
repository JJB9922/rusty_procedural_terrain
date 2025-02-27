use glium::{Surface, glutin, implement_vertex, uniform};
use nalgebra::{Matrix4, Perspective3, Point3, RealField, Vector3};
use noise::{NoiseFn, Perlin, Seedable};
use rand::Rng;
use std::collections::HashSet;
use std::sync::mpsc::channel;
use std::thread;

// Som magic (Not rly worth tweaking further unless its to perf becnh with render distance)
const GRID_SIZE: usize = 64;
const CHUNK_RENDER_DISTANCE: i32 = 20;
const SCALE: f64 = 0.03;
const HEIGHT_SCALE: f32 = 20.0;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

implement_vertex!(Vertex, position, color);

struct TerrainChunk {
    vertex_buffer: glium::VertexBuffer<Vertex>,
    index_buffer: glium::IndexBuffer<u32>,
    position: (i32, i32),
    distance_to_camera: f32,
}

fn generate_chunk_data(perlin: &Perlin, chunk_x: i32, chunk_z: i32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(GRID_SIZE * GRID_SIZE);
    let mut indices = Vec::with_capacity((GRID_SIZE - 1) * (GRID_SIZE - 1) * 6);
    let base_x = chunk_x * (GRID_SIZE as i32 - 1);
    let base_z = chunk_z * (GRID_SIZE as i32 - 1);
    for x in 0..GRID_SIZE {
        for z in 0..GRID_SIZE {
            let world_x = base_x + x as i32;
            let world_z = base_z + z as i32;
            let nx = world_x as f64 * SCALE;
            let nz = world_z as f64 * SCALE;
            let height1 = perlin.get([nx, nz]) as f32;
            let height2 = perlin.get([nx * 2.0, nz * 2.0]) as f32 * 0.5;
            let height3 = perlin.get([nx * 4.0, nz * 4.0]) as f32 * 0.25;
            let height = (height1 + height2 + height3) * HEIGHT_SCALE;

            // Should strip the magic numbers entirely but i do not care
            let color = if height < -6.0 {
                [0.0, 0.0, 0.6]
            } else if height < -2.0 {
                [0.0, 0.3, 0.7]
            } else if height < 0.0 {
                [0.1, 0.5, 0.8]
            } else if height < 1.0 {
                [0.8, 0.7, 0.5]
            } else if height < 6.0 {
                [0.2, 0.6, 0.1]
            } else if height < 12.0 {
                [0.4, 0.3, 0.2]
            } else if height < 16.0 {
                [0.6, 0.6, 0.6]
            } else {
                [1.0, 1.0, 1.0]
            };
            vertices.push(Vertex {
                position: [world_x as f32, height, world_z as f32],
                color,
            });
        }
    }
    for x in 0..GRID_SIZE - 1 {
        for z in 0..GRID_SIZE - 1 {
            let top_left = (x * GRID_SIZE + z) as u32;
            let top_right = (x * GRID_SIZE + (z + 1)) as u32;
            let bottom_left = ((x + 1) * GRID_SIZE + z) as u32;
            let bottom_right = ((x + 1) * GRID_SIZE + (z + 1)) as u32;
            indices.push(top_left);
            indices.push(bottom_left);
            indices.push(top_right);
            indices.push(top_right);
            indices.push(bottom_left);
            indices.push(bottom_right);
        }
    }
    (vertices, indices)
}

fn main() {
    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("AAAAAAAAAAAAA RUST RATIO AAAAAAAAAAAAAA")
        .with_inner_size(glutin::dpi::LogicalSize::new(1920.0, 1080.0));
    let cb = glutin::ContextBuilder::new()
        .with_depth_buffer(24)
        .with_multisampling(4);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();
    // ??
    let seed = rand::thread_rng().gen_range(0, 10000);
    let perlin = Perlin::new(seed).set_seed(seed);

    let mut camera_pos = Point3::new(0.0, 40.0, 0.0);
    let mut camera_chunk = (0, 0);
    let mut chunks: Vec<TerrainChunk> = Vec::new();
    let mut pending_chunks: HashSet<(i32, i32)> = HashSet::new();

    for radius in 1..=CHUNK_RENDER_DISTANCE {
        for x in -radius..=radius {
            for z in -radius..=radius {
                if x.abs() == radius || z.abs() == radius {
                    let (vertices, indices) = generate_chunk_data(&perlin, x, z);
                    let vb = glium::VertexBuffer::new(&display, &vertices).unwrap();
                    let ib = glium::IndexBuffer::new(
                        &display,
                        glium::index::PrimitiveType::TrianglesList,
                        &indices,
                    )
                    .unwrap();
                    chunks.push(TerrainChunk {
                        vertex_buffer: vb,
                        index_buffer: ib,
                        position: (x, z),
                        distance_to_camera: 0.0,
                    });
                }
            }
        }
    }

    let vertex_shader_src = r#"
        #version 140
        in vec3 position;
        in vec3 color;
        out vec3 v_color;
        out vec3 v_position;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            v_position = position;
            gl_Position = projection * view * model * vec4(position, 1.0);
            v_color = color;
        }
    "#;
    let fragment_shader_src = r#"
        #version 140
        in vec3 v_color;
        in vec3 v_position;
        out vec4 color;
        uniform vec3 camera_pos;
        void main() {
            vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
            float diffuse = max(0.3, dot(vec3(0.0, 1.0, 0.0), light_dir));
            float dist = length(v_position - camera_pos);
            float fog_factor = 1.0 - clamp((dist - 200.0) / 1800.0, 0.0, 1.0);
            vec3 lit_color = v_color * diffuse;
            vec3 fog_color = vec3(0.7, 0.8, 0.9);
            vec3 final_color = mix(fog_color, lit_color, fog_factor);
            color = vec4(final_color, 1.0);
        }
    "#;
    let program =
        glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None)
            .unwrap();

    let mut last_update = std::time::Instant::now();
    let mut yaw = std::f32::consts::PI;
    let mut pitch = 0.0f32;
    let mut forward = false;
    let mut backward = false;
    let mut left = false;
    let mut right = false;
    let mut up = false;
    let mut down = false;
    let mut speed_multiplier = 1.0f32;

    let draw_parameters = glium::DrawParameters {
        depth: glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLess,
            write: true,
            ..Default::default()
        },
        backface_culling: glium::draw_parameters::BackfaceCullingMode::CullingDisabled,
        ..Default::default()
    };

    let (chunk_req_sender, chunk_req_receiver) = channel::<(i32, i32)>();
    let (chunk_resp_sender, chunk_resp_receiver) = channel::<(i32, i32, Vec<Vertex>, Vec<u32>)>();
    let perlin_worker = perlin.clone();
    thread::spawn(move || {
        while let Ok((chunk_x, chunk_z)) = chunk_req_receiver.recv() {
            let data = generate_chunk_data(&perlin_worker, chunk_x, chunk_z);
            chunk_resp_sender
                .send((chunk_x, chunk_z, data.0, data.1))
                .unwrap();
        }
    });

    event_loop.run(move |event, _, control_flow| {
        let now = std::time::Instant::now();
        let delta_time = now.duration_since(last_update).as_secs_f32();
        last_update = now;
        let base_speed = 40.0 * delta_time * speed_multiplier;
        let height_factor = (camera_pos.y / 20.0).max(1.0);
        let speed = base_speed * height_factor;
        let front = Vector3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        )
        .normalize();
        let right_vec = Vector3::new(-yaw.sin(), 0.0, yaw.cos()).normalize();

        // no event bus sad :(
        if forward {
            camera_pos += front * speed;
        }
        if backward {
            camera_pos -= front * speed;
        }
        if right {
            camera_pos += right_vec * speed;
        }
        if left {
            camera_pos -= right_vec * speed;
        }
        if up {
            camera_pos.y += speed;
        }
        if down {
            camera_pos.y -= speed;
        }
        let nx = camera_pos.x as f64 * SCALE;
        let nz = camera_pos.z as f64 * SCALE;
        let ground_height = (perlin.get([nx, nz]) as f32
            + perlin.get([nx * 2.0, nz * 2.0]) as f32 * 0.5
            + perlin.get([nx * 4.0, nz * 4.0]) as f32 * 0.25)
            * HEIGHT_SCALE;
        camera_pos.y = camera_pos.y.max(ground_height + 2.0);
        let new_chunk_x = (camera_pos.x / (GRID_SIZE - 1) as f32).floor() as i32;
        let new_chunk_z = (camera_pos.z / (GRID_SIZE - 1) as f32).floor() as i32;
        if new_chunk_x != camera_chunk.0 || new_chunk_z != camera_chunk.1 {
            camera_chunk = (new_chunk_x, new_chunk_z);
            chunks.retain(|chunk| {
                let dx = chunk.position.0 - camera_chunk.0;
                let dz = chunk.position.1 - camera_chunk.1;
                dx.abs() <= CHUNK_RENDER_DISTANCE && dz.abs() <= CHUNK_RENDER_DISTANCE
            });
            for chunk_x in
                camera_chunk.0 - CHUNK_RENDER_DISTANCE..=camera_chunk.0 + CHUNK_RENDER_DISTANCE
            {
                for chunk_z in
                    camera_chunk.1 - CHUNK_RENDER_DISTANCE..=camera_chunk.1 + CHUNK_RENDER_DISTANCE
                {
                    let dist = ((chunk_x - camera_chunk.0).pow(2)
                        + (chunk_z - camera_chunk.1).pow(2)) as f32;
                    if dist <= (CHUNK_RENDER_DISTANCE * CHUNK_RENDER_DISTANCE) as f32
                        && !chunks.iter().any(|c| c.position == (chunk_x, chunk_z))
                        && !pending_chunks.contains(&(chunk_x, chunk_z))
                    {
                        chunk_req_sender.send((chunk_x, chunk_z)).unwrap();
                        pending_chunks.insert((chunk_x, chunk_z));
                    }
                }
            }
        }
        for chunk in &mut chunks {
            let chunk_center_x =
                (chunk.position.0 * (GRID_SIZE - 1) as i32) as f32 + (GRID_SIZE / 2) as f32;
            let chunk_center_z =
                (chunk.position.1 * (GRID_SIZE - 1) as i32) as f32 + (GRID_SIZE / 2) as f32;
            chunk.distance_to_camera = ((chunk_center_x - camera_pos.x).powi(2)
                + (chunk_center_z - camera_pos.z).powi(2))
            .sqrt();
        }
        chunks.sort_unstable_by(|a, b| {
            b.distance_to_camera
                .partial_cmp(&a.distance_to_camera)
                .unwrap()
        });
        for (chunk_x, chunk_z, vertices, indices) in chunk_resp_receiver.try_iter() {
            let vb = glium::VertexBuffer::new(&display, &vertices).unwrap();
            let ib = glium::IndexBuffer::new(
                &display,
                glium::index::PrimitiveType::TrianglesList,
                &indices,
            )
            .unwrap();
            chunks.push(TerrainChunk {
                vertex_buffer: vb,
                index_buffer: ib,
                position: (chunk_x, chunk_z),
                distance_to_camera: 0.0,
            });
            pending_chunks.remove(&(chunk_x, chunk_z));
        }
        let look_target = camera_pos + front;
        let view = Matrix4::look_at_rh(
            &camera_pos,
            &Point3::new(look_target.x, look_target.y, look_target.z),
            &Vector3::y_axis(),
        );
        let projection = Perspective3::new(
            display.get_framebuffer_dimensions().0 as f32
                / display.get_framebuffer_dimensions().1 as f32,
            std::f32::consts::FRAC_PI_3,
            0.1,
            5000.0,
        );
        let model = Matrix4::identity();
        let mut target = display.draw();
        // Da sky
        target.clear_color_and_depth((0.7, 0.8, 0.9, 1.0), 1.0);
        for chunk in &chunks {
            let uniforms = uniform! {
                model: Into::<[[f32; 4]; 4]>::into(model),
                view: Into::<[[f32; 4]; 4]>::into(view),
                projection: Into::<[[f32; 4]; 4]>::into(*projection.as_matrix()),
                camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z]
            };
            target
                .draw(
                    &chunk.vertex_buffer,
                    &chunk.index_buffer,
                    &program,
                    &uniforms,
                    &draw_parameters,
                )
                .unwrap();
        }
        target.finish().unwrap();
        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                }
                glutin::event::WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(key) = input.virtual_keycode {
                        let pressed = input.state == glutin::event::ElementState::Pressed;
                        match key {
                            glutin::event::VirtualKeyCode::W => forward = pressed,
                            glutin::event::VirtualKeyCode::S => backward = pressed,
                            glutin::event::VirtualKeyCode::A => left = pressed,
                            glutin::event::VirtualKeyCode::D => right = pressed,
                            glutin::event::VirtualKeyCode::Space => up = pressed,
                            glutin::event::VirtualKeyCode::LShift => down = pressed,
                            glutin::event::VirtualKeyCode::Left => {
                                if pressed {
                                    yaw += 0.05
                                }
                            }
                            glutin::event::VirtualKeyCode::Right => {
                                if pressed {
                                    yaw -= 0.05
                                }
                            }
                            glutin::event::VirtualKeyCode::Up => {
                                if pressed {
                                    pitch = (pitch + 0.05).min(std::f32::consts::FRAC_PI_2 - 0.1)
                                }
                            }
                            glutin::event::VirtualKeyCode::Down => {
                                if pressed {
                                    pitch = (pitch - 0.05).max(-std::f32::consts::FRAC_PI_2 + 0.1)
                                }
                            }
                            // Faster?
                            glutin::event::VirtualKeyCode::LControl => {
                                speed_multiplier = if pressed { 3.0 } else { 1.0 };
                            }
                            glutin::event::VirtualKeyCode::Escape => {
                                *control_flow = glutin::event_loop::ControlFlow::Exit;
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            },
            glutin::event::Event::MainEventsCleared => {
                display.gl_window().window().request_redraw();
            }
            _ => {}
        }
        *control_flow = glutin::event_loop::ControlFlow::Poll;
    });
}
