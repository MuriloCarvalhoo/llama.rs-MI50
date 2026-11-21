//! Único módulo do crate com `unsafe`: os wrappers das chamadas do `ash` (todas são
//! `unsafe fn`), o mapa de memória host-visible e a ioctl do pstate.
//!
//! **Contrato.** Os wrappers são seguros no tipo, não na semântica: o Vulkan não valida
//! uso e o `ash` não rastreia vida de objetos. Quem chama garante o que a spec pede —
//! handles criados por este device e ainda vivos, create-infos cujos ponteiros (`p_*`)
//! apontam para dados vivos durante a chamada, command buffer em gravação nos `cmd_*`, e
//! nada destruído enquanto a GPU ainda o usa (fence esperado ou `device_wait_idle`). É o
//! mesmo contrato que o resto do crate já cumpria dentro dos blocos `unsafe` espalhados;
//! aqui ele fica num lugar só. Zerar este módulo exige trocar o `ash` — ver
//! `docs/planos/2026-11-21-zerar-unsafe-vulkan.md`.
#![allow(unsafe_code)]

use ash::prelude::VkResult;
use ash::vk;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::ptr::NonNull;

/// Carrega a biblioteca Vulkan do sistema.
pub(crate) fn carregar_entry() -> Result<ash::Entry, ash::LoadingError> {
    // SAFETY: só faz dlopen da libvulkan do sistema; a `Entry` devolvida mantém a lib
    // carregada enquanto viver (o `VulkanContext` a guarda junto da instância).
    unsafe { ash::Entry::load() }
}

/// Cria a instância Vulkan.
pub(crate) fn create_instance(
    entry: &ash::Entry,
    create_info: &vk::InstanceCreateInfo<'_>,
) -> VkResult<Instance> {
    // SAFETY: `create_info` e o `ApplicationInfo` para o qual aponta vivem na pilha do
    // chamador até a chamada retornar.
    unsafe { entry.create_instance(create_info, None) }.map(Instance)
}

/// `ash::Instance` com os métodos usados pelo crate em versão segura.
pub struct Instance(ash::Instance);

/// `ash::Device` com os métodos usados pelo crate em versão segura.
pub struct Device(ash::Device);

impl Instance {
    pub(crate) fn create_device(
        &self,
        physical_device: vk::PhysicalDevice,
        create_info: &vk::DeviceCreateInfo<'_>,
    ) -> VkResult<Device> {
        // SAFETY: `physical_device` veio de `enumerate_physical_devices` desta instância;
        // `create_info` e as queue-infos apontadas vivem na pilha do chamador.
        unsafe { self.0.create_device(physical_device, create_info, None) }.map(Device)
    }
}

impl Device {
    /// Mapeia `size` bytes de `memory` a partir de `offset`. O mapa vale até o
    /// `unmap_memory` correspondente; usá-lo depois disso viola o contrato do módulo.
    pub(crate) fn map_memory(
        &self,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        flags: vk::MemoryMapFlags,
    ) -> VkResult<Mapa> {
        let len = usize::try_from(size).map_err(|_| vk::Result::ERROR_MEMORY_MAP_FAILED)?;
        // SAFETY: memória host-visible deste device, ainda não mapeada (contrato do módulo).
        let ptr = unsafe { self.0.map_memory(memory, offset, size, flags) }?;
        let ptr = NonNull::new(ptr.cast::<u8>()).ok_or(vk::Result::ERROR_MEMORY_MAP_FAILED)?;
        Ok(Mapa { ptr, len })
    }
}

/// Memória host-visible mapeada: `len` bytes a partir de `ptr`.
pub(crate) struct Mapa {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: o ponteiro é de memória mapeada no espaço do processo, sem afinidade de thread;
// o acesso continua passando por `&self`/`&mut self`.
unsafe impl Send for Mapa {}

impl Mapa {
    /// Os bytes mapeados. Quem lê garante que a GPU não escreve neles agora (fence esperado).
    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: `ptr` aponta para `len` bytes mapeados enquanto o mapa viver (contrato
        // de `map_memory`); a memória é HOST_COHERENT, sem flush/invalidate.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Os bytes mapeados, para escrita. Quem escreve garante que a GPU não os lê agora.
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: como em `bytes`; `&mut self` impede outro empréstimo do mesmo mapa.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Copia `src` para o começo do mapa.
    pub(crate) fn escrever<T: bytemuck::Pod>(&mut self, src: &[T]) {
        let src = bytemuck::cast_slice::<T, u8>(src);
        self.bytes_mut()[..src.len()].copy_from_slice(src);
    }

    /// Os `n` primeiros `T` do mapa, copiados. `to_vec` de um slice copia sem zerar antes.
    pub(crate) fn ler<T: bytemuck::Pod>(&self, n: usize) -> Vec<T> {
        bytemuck::cast_slice(&self.bytes()[..n * size_of::<T>()]).to_vec()
    }
}

// uapi/drm/amdgpu_drm.h: DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_CTX, union drm_amdgpu_ctx),
// com a union de 16 bytes.
const DRM_IOCTL_AMDGPU_CTX: libc::c_ulong = 0xC010_6442;

/// `DRM_IOCTL_AMDGPU_CTX` com a `union drm_amdgpu_ctx` em `args`.
pub(crate) fn amdgpu_ctx_ioctl(fd: &File, args: &mut [u32; 4]) -> std::io::Result<()> {
    // SAFETY: `fd` é um arquivo aberto; `args` tem os 16 bytes e o layout da union
    // `drm_amdgpu_ctx` que o número da ioctl declara, e vive durante a chamada. Num fd
    // que não é render node amdgpu o kernel só devolve erro.
    let r = unsafe {
        libc::ioctl(
            fd.as_raw_fd(),
            DRM_IOCTL_AMDGPU_CTX as _,
            std::ptr::from_mut(args),
        )
    };
    if r == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}
impl Device {
    pub(crate) fn free_memory(
        &self,
        memory: vk::DeviceMemory,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi alocado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe { self.0.free_memory(memory, allocation_callbacks) }
    }

    pub(crate) fn destroy_buffer(
        &self,
        buffer: vk::Buffer,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi criado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe { self.0.destroy_buffer(buffer, allocation_callbacks) }
    }

    pub(crate) fn unmap_memory(&self, memory: vk::DeviceMemory) {
        // SAFETY: handles válidos deste device e ponteiros vivos durante a chamada (contrato do módulo).
        unsafe { self.0.unmap_memory(memory) }
    }

    pub(crate) fn queue_submit(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> VkResult<()> {
        // SAFETY: a fila é deste device e os command buffers submetidos terminaram de gravar (contrato do módulo).
        unsafe { self.0.queue_submit(queue, submits, fence) }
    }

    pub(crate) fn end_command_buffer(&self, command_buffer: vk::CommandBuffer) -> VkResult<()> {
        // SAFETY: handles válidos deste device e ponteiros vivos durante a chamada (contrato do módulo).
        unsafe { self.0.end_command_buffer(command_buffer) }
    }

    pub(crate) fn begin_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        begin_info: &vk::CommandBufferBeginInfo<'_>,
    ) -> VkResult<()> {
        // SAFETY: handles válidos deste device e ponteiros vivos durante a chamada (contrato do módulo).
        unsafe { self.0.begin_command_buffer(command_buffer, begin_info) }
    }

    pub(crate) fn free_command_buffers(
        &self,
        command_pool: vk::CommandPool,
        command_buffers: &[vk::CommandBuffer],
    ) {
        // SAFETY: o objeto foi alocado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe { self.0.free_command_buffers(command_pool, command_buffers) }
    }

    pub(crate) fn allocate_command_buffers(
        &self,
        allocate_info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> VkResult<Vec<vk::CommandBuffer>> {
        // SAFETY: as allocate-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe { self.0.allocate_command_buffers(allocate_info) }
    }

    pub(crate) fn cmd_copy_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        // SAFETY: o command buffer está em gravação e os handles e ponteiros da chamada vêm deste device e vivem até ela retornar (contrato do módulo).
        unsafe {
            self.0
                .cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, regions)
        }
    }

    pub(crate) fn queue_wait_idle(&self, queue: vk::Queue) -> VkResult<()> {
        // SAFETY: a fila é deste device e os command buffers submetidos terminaram de gravar (contrato do módulo).
        unsafe { self.0.queue_wait_idle(queue) }
    }

    pub(crate) fn update_descriptor_sets(
        &self,
        descriptor_writes: &[vk::WriteDescriptorSet<'_>],
        descriptor_copies: &[vk::CopyDescriptorSet<'_>],
    ) {
        // SAFETY: handles válidos deste device e ponteiros vivos durante a chamada (contrato do módulo).
        unsafe {
            self.0
                .update_descriptor_sets(descriptor_writes, descriptor_copies)
        }
    }

    pub(crate) fn cmd_push_constants(
        &self,
        command_buffer: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        // SAFETY: o command buffer está em gravação e os handles e ponteiros da chamada vêm deste device e vivem até ela retornar (contrato do módulo).
        unsafe {
            self.0
                .cmd_push_constants(command_buffer, layout, stage_flags, offset, constants)
        }
    }

    pub(crate) fn cmd_dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        // SAFETY: o command buffer está em gravação e os handles e ponteiros da chamada vêm deste device e vivem até ela retornar (contrato do módulo).
        unsafe {
            self.0
                .cmd_dispatch(command_buffer, group_count_x, group_count_y, group_count_z)
        }
    }

    pub(crate) fn cmd_bind_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline_bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        // SAFETY: o command buffer está em gravação e os handles e ponteiros da chamada vêm deste device e vivem até ela retornar (contrato do módulo).
        unsafe {
            self.0
                .cmd_bind_pipeline(command_buffer, pipeline_bind_point, pipeline)
        }
    }

    pub(crate) fn cmd_bind_descriptor_sets(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline_bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        // SAFETY: o command buffer está em gravação e os handles e ponteiros da chamada vêm deste device e vivem até ela retornar (contrato do módulo).
        unsafe {
            self.0.cmd_bind_descriptor_sets(
                command_buffer,
                pipeline_bind_point,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            )
        }
    }

    pub(crate) fn destroy_descriptor_pool(
        &self,
        pool: vk::DescriptorPool,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi criado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe { self.0.destroy_descriptor_pool(pool, allocation_callbacks) }
    }

    pub(crate) fn create_descriptor_pool(
        &self,
        create_info: &vk::DescriptorPoolCreateInfo<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::DescriptorPool> {
        // SAFETY: as create-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe {
            self.0
                .create_descriptor_pool(create_info, allocation_callbacks)
        }
    }

    pub(crate) fn allocate_descriptor_sets(
        &self,
        allocate_info: &vk::DescriptorSetAllocateInfo<'_>,
    ) -> VkResult<Vec<vk::DescriptorSet>> {
        // SAFETY: as allocate-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe { self.0.allocate_descriptor_sets(allocate_info) }
    }

    pub(crate) fn reset_fences(&self, fences: &[vk::Fence]) -> VkResult<()> {
        // SAFETY: handles válidos deste device e ponteiros vivos durante a chamada (contrato do módulo).
        unsafe { self.0.reset_fences(fences) }
    }

    pub(crate) fn reset_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        flags: vk::CommandBufferResetFlags,
    ) -> VkResult<()> {
        // SAFETY: handles válidos deste device e ponteiros vivos durante a chamada (contrato do módulo).
        unsafe { self.0.reset_command_buffer(command_buffer, flags) }
    }

    pub(crate) fn device_wait_idle(&self) -> VkResult<()> {
        // SAFETY: handles válidos deste device e ponteiros vivos durante a chamada (contrato do módulo).
        unsafe { self.0.device_wait_idle() }
    }

    pub(crate) fn cmd_write_timestamp(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline_stage: vk::PipelineStageFlags,
        query_pool: vk::QueryPool,
        query: u32,
    ) {
        // SAFETY: o command buffer está em gravação e os handles e ponteiros da chamada vêm deste device e vivem até ela retornar (contrato do módulo).
        unsafe {
            self.0
                .cmd_write_timestamp(command_buffer, pipeline_stage, query_pool, query)
        }
    }

    pub(crate) fn wait_for_fences(
        &self,
        fences: &[vk::Fence],
        wait_all: bool,
        timeout: u64,
    ) -> VkResult<()> {
        // SAFETY: handles válidos deste device e ponteiros vivos durante a chamada (contrato do módulo).
        unsafe { self.0.wait_for_fences(fences, wait_all, timeout) }
    }

    pub(crate) fn destroy_pipeline_layout(
        &self,
        pipeline_layout: vk::PipelineLayout,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi criado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe {
            self.0
                .destroy_pipeline_layout(pipeline_layout, allocation_callbacks)
        }
    }

    pub(crate) fn destroy_pipeline(
        &self,
        pipeline: vk::Pipeline,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi criado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe { self.0.destroy_pipeline(pipeline, allocation_callbacks) }
    }

    pub(crate) fn destroy_fence(
        &self,
        fence: vk::Fence,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi criado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe { self.0.destroy_fence(fence, allocation_callbacks) }
    }

    pub(crate) fn destroy_descriptor_set_layout(
        &self,
        layout: vk::DescriptorSetLayout,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi criado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe {
            self.0
                .destroy_descriptor_set_layout(layout, allocation_callbacks)
        }
    }

    pub(crate) fn create_fence(
        &self,
        create_info: &vk::FenceCreateInfo<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::Fence> {
        // SAFETY: as create-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe { self.0.create_fence(create_info, allocation_callbacks) }
    }

    pub(crate) fn get_query_pool_results<T>(
        &self,
        query_pool: vk::QueryPool,
        first_query: u32,
        data: &mut [T],
        flags: vk::QueryResultFlags,
    ) -> VkResult<()> {
        // SAFETY: handles válidos deste device/instância; a saída é escrita em memória do chamador (contrato do módulo).
        unsafe {
            self.0
                .get_query_pool_results(query_pool, first_query, data, flags)
        }
    }

    pub(crate) fn get_buffer_memory_requirements(
        &self,
        buffer: vk::Buffer,
    ) -> vk::MemoryRequirements {
        // SAFETY: handles válidos deste device/instância; a saída é escrita em memória do chamador (contrato do módulo).
        unsafe { self.0.get_buffer_memory_requirements(buffer) }
    }

    pub(crate) fn destroy_query_pool(
        &self,
        pool: vk::QueryPool,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi criado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe { self.0.destroy_query_pool(pool, allocation_callbacks) }
    }

    pub(crate) fn destroy_device(
        &self,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi criado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe { self.0.destroy_device(allocation_callbacks) }
    }

    pub(crate) fn create_query_pool(
        &self,
        create_info: &vk::QueryPoolCreateInfo<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::QueryPool> {
        // SAFETY: as create-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe { self.0.create_query_pool(create_info, allocation_callbacks) }
    }

    pub(crate) fn cmd_reset_query_pool(
        &self,
        command_buffer: vk::CommandBuffer,
        pool: vk::QueryPool,
        first_query: u32,
        query_count: u32,
    ) {
        // SAFETY: o command buffer está em gravação e os handles e ponteiros da chamada vêm deste device e vivem até ela retornar (contrato do módulo).
        unsafe {
            self.0
                .cmd_reset_query_pool(command_buffer, pool, first_query, query_count)
        }
    }

    #[allow(clippy::too_many_arguments)] // espelha a assinatura do ash
    pub(crate) fn cmd_pipeline_barrier(
        &self,
        command_buffer: vk::CommandBuffer,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier<'_>],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier<'_>],
        image_memory_barriers: &[vk::ImageMemoryBarrier<'_>],
    ) {
        // SAFETY: o command buffer está em gravação e os handles e ponteiros da chamada vêm deste device e vivem até ela retornar (contrato do módulo).
        unsafe {
            self.0.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask,
                dst_stage_mask,
                dependency_flags,
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            )
        }
    }

    pub(crate) fn bind_buffer_memory(
        &self,
        buffer: vk::Buffer,
        device_memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
    ) -> VkResult<()> {
        // SAFETY: handles válidos deste device e ponteiros vivos durante a chamada (contrato do módulo).
        unsafe { self.0.bind_buffer_memory(buffer, device_memory, offset) }
    }

    pub(crate) fn allocate_memory(
        &self,
        allocate_info: &vk::MemoryAllocateInfo<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::DeviceMemory> {
        // SAFETY: as allocate-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe { self.0.allocate_memory(allocate_info, allocation_callbacks) }
    }

    pub(crate) fn get_fence_status(&self, fence: vk::Fence) -> VkResult<bool> {
        // SAFETY: handles válidos deste device/instância; a saída é escrita em memória do chamador (contrato do módulo).
        unsafe { self.0.get_fence_status(fence) }
    }

    pub(crate) fn get_device_queue(&self, queue_family_index: u32, queue_index: u32) -> vk::Queue {
        // SAFETY: handles válidos deste device/instância; a saída é escrita em memória do chamador (contrato do módulo).
        unsafe { self.0.get_device_queue(queue_family_index, queue_index) }
    }

    pub(crate) fn destroy_shader_module(
        &self,
        shader: vk::ShaderModule,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi criado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe { self.0.destroy_shader_module(shader, allocation_callbacks) }
    }

    pub(crate) fn destroy_command_pool(
        &self,
        pool: vk::CommandPool,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi criado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe { self.0.destroy_command_pool(pool, allocation_callbacks) }
    }

    pub(crate) fn create_shader_module(
        &self,
        create_info: &vk::ShaderModuleCreateInfo<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::ShaderModule> {
        // SAFETY: as create-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe {
            self.0
                .create_shader_module(create_info, allocation_callbacks)
        }
    }

    pub(crate) fn create_pipeline_layout(
        &self,
        create_info: &vk::PipelineLayoutCreateInfo<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::PipelineLayout> {
        // SAFETY: as create-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe {
            self.0
                .create_pipeline_layout(create_info, allocation_callbacks)
        }
    }

    pub(crate) fn create_descriptor_set_layout(
        &self,
        create_info: &vk::DescriptorSetLayoutCreateInfo<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::DescriptorSetLayout> {
        // SAFETY: as create-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe {
            self.0
                .create_descriptor_set_layout(create_info, allocation_callbacks)
        }
    }

    pub(crate) fn create_compute_pipelines(
        &self,
        pipeline_cache: vk::PipelineCache,
        create_infos: &[vk::ComputePipelineCreateInfo<'_>],
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)> {
        // SAFETY: as create-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe {
            self.0
                .create_compute_pipelines(pipeline_cache, create_infos, allocation_callbacks)
        }
    }

    pub(crate) fn create_command_pool(
        &self,
        create_info: &vk::CommandPoolCreateInfo<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::CommandPool> {
        // SAFETY: as create-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe {
            self.0
                .create_command_pool(create_info, allocation_callbacks)
        }
    }

    pub(crate) fn create_buffer(
        &self,
        create_info: &vk::BufferCreateInfo<'_>,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) -> VkResult<vk::Buffer> {
        // SAFETY: as create-infos e o que elas apontam vivem na pilha do chamador durante a chamada (contrato do módulo).
        unsafe { self.0.create_buffer(create_info, allocation_callbacks) }
    }

    pub(crate) fn cmd_fill_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        data: u32,
    ) {
        // SAFETY: o command buffer está em gravação e os handles e ponteiros da chamada vêm deste device e vivem até ela retornar (contrato do módulo).
        unsafe {
            self.0
                .cmd_fill_buffer(command_buffer, buffer, offset, size, data)
        }
    }
}

impl Instance {
    pub(crate) fn get_physical_device_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceProperties {
        // SAFETY: handles válidos deste device/instância; a saída é escrita em memória do chamador (contrato do módulo).
        unsafe { self.0.get_physical_device_properties(physical_device) }
    }

    pub(crate) fn get_physical_device_memory_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceMemoryProperties {
        // SAFETY: handles válidos deste device/instância; a saída é escrita em memória do chamador (contrato do módulo).
        unsafe {
            self.0
                .get_physical_device_memory_properties(physical_device)
        }
    }

    pub(crate) fn enumerate_device_extension_properties(
        &self,
        device: vk::PhysicalDevice,
    ) -> VkResult<Vec<vk::ExtensionProperties>> {
        // SAFETY: handles válidos deste device e ponteiros vivos durante a chamada (contrato do módulo).
        unsafe { self.0.enumerate_device_extension_properties(device) }
    }

    pub(crate) fn get_physical_device_queue_family_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        // SAFETY: handles válidos deste device/instância; a saída é escrita em memória do chamador (contrato do módulo).
        unsafe {
            self.0
                .get_physical_device_queue_family_properties(physical_device)
        }
    }

    pub(crate) fn get_physical_device_properties2(
        &self,
        physical_device: vk::PhysicalDevice,
        prop: &mut vk::PhysicalDeviceProperties2<'_>,
    ) {
        // SAFETY: handles válidos deste device/instância; a saída é escrita em memória do chamador (contrato do módulo).
        unsafe {
            self.0
                .get_physical_device_properties2(physical_device, prop)
        }
    }

    pub(crate) fn get_physical_device_memory_properties2(
        &self,
        physical_device: vk::PhysicalDevice,
        out: &mut vk::PhysicalDeviceMemoryProperties2<'_>,
    ) {
        // SAFETY: handles válidos deste device/instância; a saída é escrita em memória do chamador (contrato do módulo).
        unsafe {
            self.0
                .get_physical_device_memory_properties2(physical_device, out)
        }
    }

    pub(crate) fn enumerate_physical_devices(&self) -> VkResult<Vec<vk::PhysicalDevice>> {
        // SAFETY: handles válidos deste device e ponteiros vivos durante a chamada (contrato do módulo).
        unsafe { self.0.enumerate_physical_devices() }
    }

    pub(crate) fn destroy_instance(
        &self,
        allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
    ) {
        // SAFETY: o objeto foi criado por este device e a GPU não o usa mais (contrato do módulo).
        unsafe { self.0.destroy_instance(allocation_callbacks) }
    }
}
