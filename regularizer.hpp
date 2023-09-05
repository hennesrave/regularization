#pragma once
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace util
{
    uint32_t findMemoryTypeIndex( vk::PhysicalDevice physicalDevice, vk::MemoryPropertyFlags memoryPropertyFlags )
    {
        const auto memoryProperties { physicalDevice.getMemoryProperties() };

        for( uint32_t i { 0 }; i < memoryProperties.memoryTypeCount; ++i )
        {
            if( ( memoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags ) == memoryPropertyFlags )
            {
                return i;
            }
        }

        throw std::runtime_error( "Failed to find suitable memory type index" );
    }

    vk::ShaderModule createShaderModule( vk::Device device, const std::filesystem::path& filepath )
    {
        std::ifstream stream { filepath, std::ios::in | std::ios::binary };
        std::vector<char> shaderCode { ( std::istreambuf_iterator<char>( stream ) ), std::istreambuf_iterator<char>{} };

        const auto shaderModuleCreateInfo = vk::ShaderModuleCreateInfo {
            vk::ShaderModuleCreateFlags{},
            shaderCode.size(),
            reinterpret_cast<uint32_t*>( shaderCode.data() ),
            nullptr
        };
        const vk::ShaderModule shaderModule { device.createShaderModule( shaderModuleCreateInfo ) };
        return shaderModule;
    }

    class ImageWrapper
    {
    public:
        void create( vk::Device device, vk::PhysicalDevice physicalDevice, const vk::ImageCreateInfo& imageCreateInfo, vk::ImageViewCreateInfo& imageViewCreateInfo )
        {
            this->destroy();

            _device = device;
            _image = device.createImage( imageCreateInfo );

            const auto memoryRequirements { device.getImageMemoryRequirements( _image ) };
            vk::MemoryPropertyFlags memoryPropertyFlags { vk::MemoryPropertyFlagBits::eDeviceLocal };
            const uint32_t memoryTypeIndex { util::findMemoryTypeIndex( physicalDevice, memoryPropertyFlags ) };

            const auto allocateInfo = vk::MemoryAllocateInfo {
                memoryRequirements.size,
                memoryTypeIndex,
                nullptr
            };
            _memory = device.allocateMemory( allocateInfo );
            device.bindImageMemory( _image, _memory, 0 );

            imageViewCreateInfo.image = _image;
            _imageView = device.createImageView( imageViewCreateInfo );

            _format = imageViewCreateInfo.format;
        }
        void createColorImage( vk::Device device, vk::PhysicalDevice physicalDevice, vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage )
        {
            const vk::ImageType imageType { extent.depth == 1 ? ( extent.height == 1 ? ( vk::ImageType::e1D ) : ( vk::ImageType::e2D ) ) : ( vk::ImageType::e3D ) };

            const auto imageCreateInfo = vk::ImageCreateInfo {
                vk::ImageCreateFlags{},
                imageType,
                format,
                extent,
                1,
                1,
                vk::SampleCountFlagBits::e1,
                vk::ImageTiling::eOptimal,
                usage,
                vk::SharingMode::eExclusive,
                nullptr,
                vk::ImageLayout::eUndefined,
                nullptr
            };

            const vk::ImageViewType viewType {
                imageType == vk::ImageType::e1D ? ( vk::ImageViewType::e1D ) : ( imageType == vk::ImageType::e2D ? ( vk::ImageViewType::e2D ) : ( vk::ImageViewType::e3D ) )
            };

            auto imageViewCreateInfo = vk::ImageViewCreateInfo {
                vk::ImageViewCreateFlags{},
                vk::Image{},
                viewType,
                format,
                vk::ComponentMapping{},
                vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 },
                nullptr
            };

            this->create( device, physicalDevice, imageCreateInfo, imageViewCreateInfo );
        }

        void destroy()
        {
            if( _device )
            {
                _device.destroyImage( _image );
                _device.freeMemory( _memory );
                _device.destroyImageView( _imageView );

                _device = vk::Device {};
                _image = vk::Image {};
                _memory = vk::DeviceMemory {};
                _imageView = vk::ImageView {};
            }
        }

        operator bool() const noexcept
        {
            return static_cast<bool>( _device );
        }

        vk::Device device() const noexcept
        {
            return _device;
        }
        vk::Image image() const noexcept
        {
            return _image;
        }
        vk::DeviceMemory memory() const noexcept
        {
            return _memory;
        }
        vk::ImageView imageView() const noexcept
        {
            return _imageView;
        }

        vk::Format format() const noexcept
        {
            return _format;
        }

    private:
        vk::Device _device {};
        vk::Image _image {};
        vk::DeviceMemory _memory {};
        vk::ImageView _imageView {};

        vk::Format _format {};
    };

    class BufferWrapper
    {
    public:
        BufferWrapper() noexcept = default;
        BufferWrapper( vk::Device device, vk::PhysicalDevice physicalDevice, const vk::BufferCreateInfo& bufferCreateInfo, vk::MemoryPropertyFlags memoryPropertyFlags )
        {
            this->create( device, physicalDevice, bufferCreateInfo, memoryPropertyFlags );
        }
        BufferWrapper( vk::Device device, vk::PhysicalDevice physicalDevice, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryPropertyFlags )
        {
            this->create( device, physicalDevice, size, usage, memoryPropertyFlags );
        }

        void create( vk::Device device, vk::PhysicalDevice physicalDevice, const vk::BufferCreateInfo& bufferCreateInfo, vk::MemoryPropertyFlags memoryPropertyFlags )
        {
            this->destroy();

            _device = device;
            _buffer = device.createBuffer( bufferCreateInfo );

            const auto memoryRequirements { device.getBufferMemoryRequirements( _buffer ) };
            const uint32_t memoryTypeIndex { util::findMemoryTypeIndex( physicalDevice, memoryPropertyFlags ) };

            const auto allocateInfo = vk::MemoryAllocateInfo {
                memoryRequirements.size,
                memoryTypeIndex,
                nullptr
            };
            _memory = device.allocateMemory( allocateInfo );
            device.bindBufferMemory( _buffer, _memory, 0 );
        }
        void create( vk::Device device, vk::PhysicalDevice physicalDevice, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryPropertyFlags )
        {
            const auto bufferCreateInfo = vk::BufferCreateInfo {
                vk::BufferCreateFlags{},
                size,
                usage,
                vk::SharingMode::eExclusive,
                nullptr,
                nullptr
            };
            this->create( device, physicalDevice, bufferCreateInfo, memoryPropertyFlags );
        }

        void destroy()
        {
            if( _device )
            {
                _device.destroyBuffer( _buffer );
                _device.freeMemory( _memory );

                _device = vk::Device {};
                _buffer = vk::Buffer {};
                _memory = vk::DeviceMemory {};
            }
        }

        operator bool() const noexcept
        {
            return static_cast<bool>( _device );
        }

        vk::Device device() const noexcept
        {
            return _device;
        }
        vk::Buffer buffer() const noexcept
        {
            return _buffer;
        }
        vk::DeviceMemory memory() const noexcept
        {
            return _memory;
        }

        template<class T> T* mapMemory()
        {
            void* pointer { _device.mapMemory( _memory, 0, VK_WHOLE_SIZE ) };
            return reinterpret_cast<T*>( pointer );
        }
        void unmapMemory()
        {
            _device.unmapMemory( _memory );
        }

    private:
        vk::Device _device {};
        vk::Buffer _buffer {};
        vk::DeviceMemory _memory {};
    };
}

class Regularizer
{
public:
    struct Point
    {
        float x {};
        float y {};
    };

    static constexpr uint32_t TextureSize { 1024 }; // NOTE: Shaders have to be adjusted when changing the texture size

    Regularizer()
    {
        // --- Create instance --- //
        const auto applicationInfo = vk::ApplicationInfo {
            "regularizer",
            VK_MAKE_VERSION( 1, 0, 0 ),
            nullptr,
            0,
            VK_API_VERSION_1_3,
            nullptr
        };
        const std::vector<const char*> enabledLayerNames {
            "VK_LAYER_KHRONOS_validation"
        };
        const auto instanceCreateInfo = vk::InstanceCreateInfo {
            vk::InstanceCreateFlags{},
            &applicationInfo,
            enabledLayerNames,
            nullptr,
            nullptr
        };
        _instance = vk::createInstance( instanceCreateInfo );

        // --- Pick physical device --- //
        const vk::QueueFlags queueFlags { vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute };

        const auto physicalDevices { _instance.enumeratePhysicalDevices() };
        for( const auto physicalDevice : physicalDevices )
        {
            const auto queueFamilyPropertiesVector { physicalDevice.getQueueFamilyProperties() };

            for( uint32_t queueFamilyIndex { 0 }; queueFamilyIndex < queueFamilyPropertiesVector.size(); ++queueFamilyIndex )
            {
                const auto& queueFamilyProperties { queueFamilyPropertiesVector[queueFamilyIndex] };
                if( ( queueFamilyProperties.queueFlags & queueFlags ) == queueFlags && ( queueFamilyProperties.timestampValidBits ) )
                {
                    _physicalDevice = physicalDevice;
                    _queueFamilyIndex = queueFamilyIndex;
                    _timestampValidBits = queueFamilyProperties.timestampValidBits;
                    break;
                }
            }

            if( _physicalDevice )
            {
                break;
            }
        }
        if( !_physicalDevice )
        {
            throw std::runtime_error( "Failed to find suitable physical device" );
        }

        // --- Create device --- //
        const auto resetFeatures = vk::PhysicalDeviceHostQueryResetFeatures { true, nullptr };
        const float queuePriority { 1.0f };
        const auto deviceQueueCreateInfo = vk::DeviceQueueCreateInfo {
            vk::DeviceQueueCreateFlags{},
            _queueFamilyIndex,
            1,
            &queuePriority,
            nullptr
        };
        const auto deviceCreateInfo = vk::DeviceCreateInfo {
            vk::DeviceCreateFlags{},
            deviceQueueCreateInfo,
            nullptr,
            nullptr,
            nullptr,
            &resetFeatures
        };
        _device = _physicalDevice.createDevice( deviceCreateInfo );
        _queue = _device.getQueue( _queueFamilyIndex, 0 );

        // --- Create command pool --- //
        const auto commandPoolCreateInfo = vk::CommandPoolCreateInfo {
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            _queueFamilyIndex,
            nullptr
        };
        _commandPool = _device.createCommandPool( commandPoolCreateInfo );

        // --- Allocate command buffer --- //
        const auto commandBufferAllocateInfo = vk::CommandBufferAllocateInfo {
            _commandPool,
            vk::CommandBufferLevel::ePrimary,
            1,
            nullptr
        };
        _commandBuffer = _device.allocateCommandBuffers( commandBufferAllocateInfo ).front();

        // --- Create query pool --- //
        const auto queryPoolCreateInfo = vk::QueryPoolCreateInfo {
            vk::QueryPoolCreateFlags{},
            vk::QueryType::eTimestamp,
            2,
            vk::QueryPipelineStatisticFlags{},
            nullptr
        };
        _queryPool = _device.createQueryPool( queryPoolCreateInfo );

        // --- Create pipeline cache --- //
        const auto pipelineCacheCreateInfo = vk::PipelineCacheCreateInfo {};
        _pipelineCache = _device.createPipelineCache( pipelineCacheCreateInfo );

        // --- Create images --- //
        this->createImages();

        // --- Create render pass --- //
        const auto attachments = std::vector<vk::AttachmentDescription> {
            vk::AttachmentDescription { vk::AttachmentDescriptionFlags{},
                _images.density.format(),
                vk::SampleCountFlagBits::e1,
                vk::AttachmentLoadOp::eClear,
                vk::AttachmentStoreOp::eStore,
                vk::AttachmentLoadOp::eDontCare,
                vk::AttachmentStoreOp::eDontCare,
                vk::ImageLayout::eGeneral,
                vk::ImageLayout::eGeneral
            }
        };
        const auto colorAttachments = std::vector<vk::AttachmentReference> {
            vk::AttachmentReference{ 0, vk::ImageLayout::eColorAttachmentOptimal }
        };
        const auto subpasses = std::vector<vk::SubpassDescription> {
            vk::SubpassDescription{ vk::SubpassDescriptionFlags{}, vk::PipelineBindPoint::eGraphics, nullptr, colorAttachments, nullptr, nullptr, nullptr }
        };
        const auto subpassDependency = vk::SubpassDependency {
            0,
            VK_SUBPASS_EXTERNAL,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::AccessFlagBits::eColorAttachmentWrite,
            vk::AccessFlagBits::eShaderRead,
            vk::DependencyFlagBits::eByRegion
        };
        const auto renderPassCreateInfo = vk::RenderPassCreateInfo {
            vk::RenderPassCreateFlags{},
            attachments,
            subpasses,
            subpassDependency,
            nullptr
        };
        _renderPass = _device.createRenderPass( renderPassCreateInfo );

        // --- Create framebuffer --- //
        const auto framebufferAttachments = std::vector<vk::ImageView> { _images.density.imageView() };
        const auto framebufferCreateInfo = vk::FramebufferCreateInfo {
            vk::FramebufferCreateFlags{},
            _renderPass,
            framebufferAttachments,
            TextureSize,
            TextureSize,
            1,
            nullptr
        };
        _framebuffer = _device.createFramebuffer( framebufferCreateInfo );

        // --- Create sampler --- //
        const auto samplerCreateInfo = vk::SamplerCreateInfo {
            vk::SamplerCreateFlags{},
            vk::Filter::eNearest,
            vk::Filter::eNearest,
            vk::SamplerMipmapMode::eNearest,
            vk::SamplerAddressMode::eClampToBorder,
            vk::SamplerAddressMode::eClampToBorder,
            vk::SamplerAddressMode::eClampToBorder,
            0.0f,
            false,
            0.0f,
            false,
            vk::CompareOp::eNever,
            0.0f,
            1.0f,
            vk::BorderColor::eFloatTransparentBlack,
            false,
            nullptr
        };
        _sampler = _device.createSampler( samplerCreateInfo );

        // --- Create descriptor set layout --- //
        const auto bindings = std::vector<vk::DescriptorSetLayoutBinding> {
            vk::DescriptorSetLayoutBinding { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eCompute, nullptr },     // points
            vk::DescriptorSetLayoutBinding { 1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr },                                         // density image
            vk::DescriptorSetLayoutBinding { 2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr },                                         // integral columns
            vk::DescriptorSetLayoutBinding { 3, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr },                                         // integral image
            vk::DescriptorSetLayoutBinding { 4, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr },                                         // upper-left integral triangle
            vk::DescriptorSetLayoutBinding { 5, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr },                                         // upper-right integral triangle
            vk::DescriptorSetLayoutBinding { 6, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr },                                         // deformation
        };
        const auto descriptorSetLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo { vk::DescriptorSetLayoutCreateFlags{}, bindings };
        _descriptorSetLayout = _device.createDescriptorSetLayout( descriptorSetLayoutCreateInfo );
        // --- Create descriptor pool --- //
        const auto descriptorPoolSizes = std::vector<vk::DescriptorPoolSize> {
            vk::DescriptorPoolSize { vk::DescriptorType::eStorageBuffer, 1 },   // points
            vk::DescriptorPoolSize { vk::DescriptorType::eStorageImage, 6 }     // density + integral columns + integral image + upper-left integral triangle + upper-right integral triangle + deformation
        };
        const auto descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo { vk::DescriptorPoolCreateFlags{}, 2, descriptorPoolSizes, nullptr };
        _descriptorPool = _device.createDescriptorPool( descriptorPoolCreateInfo );

        // --- Allocate descriptor set --- //        
        const auto descriptorSetAllocateInfo = vk::DescriptorSetAllocateInfo { _descriptorPool, _descriptorSetLayout, nullptr };
        _descriptorSet = _device.allocateDescriptorSets( descriptorSetAllocateInfo ).front();

        // --- Create pipeline layout --- //
        const auto pushConstantRanges = std::vector<vk::PushConstantRange> {
            vk::PushConstantRange { vk::ShaderStageFlagBits::eCompute, 0, sizeof( PushConstants ) }
        };
        const auto pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo { vk::PipelineLayoutCreateFlags{}, _descriptorSetLayout, pushConstantRanges, nullptr };
        _pipelineLayout = _device.createPipelineLayout( pipelineLayoutCreateInfo );

        // --- Create pipelines --- //
        this->createPipelines();
    }
    ~Regularizer()
    {
        _device.destroyPipeline( _pipelines.regularization );
        _device.destroyPipeline( _pipelines.deformation );
        _device.destroyPipeline( _pipelines.upperRightIntegralTriangle );
        _device.destroyPipeline( _pipelines.upperLeftIntegralTriangle );
        _device.destroyPipeline( _pipelines.integralImage );
        _device.destroyPipeline( _pipelines.integralColumns );
        _device.destroyPipeline( _pipelines.gaussianKernelY );
        _device.destroyPipeline( _pipelines.gaussianKernelX );
        _device.destroyPipeline( _pipelines.accumulation );

        _device.destroyPipelineLayout( _pipelineLayout );
        _device.destroyDescriptorPool( _descriptorPool );
        _device.destroyDescriptorSetLayout( _descriptorSetLayout );

        _device.destroySampler( _sampler );

        _device.destroyFramebuffer( _framebuffer );
        _device.destroyRenderPass( _renderPass );

        _images.density.destroy();
        _images.integralColumns.destroy();
        _images.integralImage.destroy();
        _images.upperLeftIntegralTriangle.destroy();
        _images.upperRightIntegralTriangle.destroy();
        _images.deformation.destroy();

        _device.destroyPipelineCache( _pipelineCache );
        _device.destroyQueryPool( _queryPool );
        _device.destroyCommandPool( _commandPool );

        _device.destroy();
        _instance.destroy();
    }

    util::BufferWrapper uploadPoints( const std::vector<Point>& points )
    {
        const vk::DeviceSize bufferSize { points.size() * sizeof( Regularizer::Point ) };

        // --- Copy points to host buffer --- //
        util::BufferWrapper pointsHost { _device, _physicalDevice, bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        };
        Regularizer::Point* memory { pointsHost.mapMemory<Regularizer::Point>() };
        std::memcpy( memory, points.data(), bufferSize );
        pointsHost.unmapMemory();

        // --- Copy points from host to device buffer --- //
        util::BufferWrapper pointsDevice { _device, _physicalDevice, bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        };

        _commandBuffer.reset();
        _commandBuffer.begin( vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit } );
        _commandBuffer.copyBuffer( pointsHost.buffer(), pointsDevice.buffer(), vk::BufferCopy{ 0, 0, bufferSize } );
        _commandBuffer.end();

        _queue.submit( vk::SubmitInfo{ nullptr, nullptr, _commandBuffer, nullptr } );
        _queue.waitIdle();

        pointsHost.destroy();
        return pointsDevice;
    }
    std::vector<Point> downloadPoints( util::BufferWrapper pointsBuffer, uint32_t pointCount )
    {
        const vk::DeviceSize bufferSize { pointCount * sizeof( Regularizer::Point ) };

        // --- Copy points from device to host buffer --- //
        util::BufferWrapper pointsHost { _device, _physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent };

        _commandBuffer.reset();
        _commandBuffer.begin( vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit } );
        _commandBuffer.copyBuffer( pointsBuffer.buffer(), pointsHost.buffer(), vk::BufferCopy{ 0, 0, bufferSize } );
        _commandBuffer.end();

        _queue.submit( vk::SubmitInfo{ nullptr, nullptr, _commandBuffer, nullptr } );
        _queue.waitIdle();

        // --- Copy points from host buffer --- //
        std::vector<Point> points( pointCount );

        Regularizer::Point* memory { pointsHost.mapMemory<Regularizer::Point>() };
        std::memcpy( points.data(), memory, bufferSize );
        pointsHost.unmapMemory();

        pointsHost.destroy();
        return points;
    }

    double regularize( vk::Buffer pointsBuffer, uint32_t pointCount, uint32_t kernelRadius, uint32_t iterations )
    {
        // --- Update descriptor sets --- //
        const auto descriptorBufferInfos = std::vector<vk::DescriptorBufferInfo> {
            vk::DescriptorBufferInfo { pointsBuffer, 0, VK_WHOLE_SIZE }
        };
        const auto descriptorImageInfos = std::vector<vk::DescriptorImageInfo> {
            vk::DescriptorImageInfo { _sampler, _images.density.imageView(), vk::ImageLayout::eGeneral },
            vk::DescriptorImageInfo { _sampler, _images.integralColumns.imageView(), vk::ImageLayout::eGeneral },
            vk::DescriptorImageInfo { _sampler, _images.integralImage.imageView(), vk::ImageLayout::eGeneral },
            vk::DescriptorImageInfo { _sampler, _images.upperLeftIntegralTriangle.imageView(), vk::ImageLayout::eGeneral },
            vk::DescriptorImageInfo { _sampler, _images.upperRightIntegralTriangle.imageView(), vk::ImageLayout::eGeneral },
            vk::DescriptorImageInfo { _sampler, _images.deformation.imageView(), vk::ImageLayout::eGeneral }
        };
        const auto descriptorWrites = std::vector<vk::WriteDescriptorSet> {
            vk::WriteDescriptorSet { _descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr,& descriptorBufferInfos[0], nullptr, nullptr },
            vk::WriteDescriptorSet { _descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageImage,& descriptorImageInfos[0], nullptr, nullptr, nullptr },
            vk::WriteDescriptorSet { _descriptorSet, 2, 0, 1, vk::DescriptorType::eStorageImage,& descriptorImageInfos[1], nullptr, nullptr, nullptr },
            vk::WriteDescriptorSet { _descriptorSet, 3, 0, 1, vk::DescriptorType::eStorageImage,& descriptorImageInfos[2], nullptr, nullptr, nullptr },
            vk::WriteDescriptorSet { _descriptorSet, 4, 0, 1, vk::DescriptorType::eStorageImage,& descriptorImageInfos[3], nullptr, nullptr, nullptr },
            vk::WriteDescriptorSet { _descriptorSet, 5, 0, 1, vk::DescriptorType::eStorageImage,& descriptorImageInfos[4], nullptr, nullptr, nullptr },
            vk::WriteDescriptorSet { _descriptorSet, 6, 0, 1, vk::DescriptorType::eStorageImage,& descriptorImageInfos[5], nullptr, nullptr, nullptr }
        };
        _device.updateDescriptorSets( descriptorWrites, nullptr );

        // --- Reset timestamp queries --- //
        _device.resetQueryPool( _queryPool, 0, 2 );

        // --- Prepare push constants --- //
        const auto pushConstants = PushConstants { pointCount, kernelRadius };

        // --- Perform regularization --- //
        _commandBuffer.reset();
        _commandBuffer.begin( vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit } );
        _commandBuffer.writeTimestamp( vk::PipelineStageFlagBits::eTopOfPipe, _queryPool, 0 );

        for( uint32_t i { 0 }; i < iterations; ++i )
        {
            // --- Compute density image --- //
            const float constantDensity { static_cast<float>( pointCount ) / ( TextureSize * TextureSize ) };
            const vk::Rect2D renderArea { vk::Offset2D{ 0, 0 }, vk::Extent2D{ TextureSize, TextureSize } };
            const vk::ClearValue clearValues { vk::ClearColorValue{ std::array<float, 4> { constantDensity, 0.0f, 0.0f, 0.0f } } };
            const auto renderPassBeginInfo = vk::RenderPassBeginInfo { _renderPass, _framebuffer, renderArea, clearValues, nullptr };
            _commandBuffer.beginRenderPass( renderPassBeginInfo, vk::SubpassContents::eInline );
            _commandBuffer.bindPipeline( vk::PipelineBindPoint::eGraphics, _pipelines.accumulation );
            _commandBuffer.bindDescriptorSets( vk::PipelineBindPoint::eGraphics, _pipelineLayout, 0, _descriptorSet, nullptr );
            _commandBuffer.draw( pointCount, 1, 0, 0 );
            _commandBuffer.endRenderPass();

            // --- Gaussian kernel x --- //
            _commandBuffer.bindPipeline( vk::PipelineBindPoint::eCompute, _pipelines.gaussianKernelX );
            _commandBuffer.bindDescriptorSets( vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, _descriptorSet, nullptr );
            _commandBuffer.pushConstants( _pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof( PushConstants ), &pushConstants );
            _commandBuffer.dispatch( TextureSize, 1, 1 );

            // --- Gaussian kernel y --- //
            auto memoryBarrier = vk::MemoryBarrier { vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead };
            _commandBuffer.pipelineBarrier( vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits::eByRegion, memoryBarrier, nullptr, nullptr );

            _commandBuffer.bindPipeline( vk::PipelineBindPoint::eCompute, _pipelines.gaussianKernelY );
            _commandBuffer.bindDescriptorSets( vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, _descriptorSet, nullptr );
            _commandBuffer.pushConstants( _pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof( PushConstants ), &pushConstants );
            _commandBuffer.dispatch( TextureSize, 1, 1 );

            // --- Compute integral columns --- //
            memoryBarrier = vk::MemoryBarrier { vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead };
            _commandBuffer.pipelineBarrier( vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits::eByRegion, memoryBarrier, nullptr, nullptr );

            _commandBuffer.bindPipeline( vk::PipelineBindPoint::eCompute, _pipelines.integralColumns );
            _commandBuffer.bindDescriptorSets( vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, _descriptorSet, nullptr );
            _commandBuffer.dispatch( TextureSize, 1, 1 );

            // --- Compute integral image --- //
            memoryBarrier = vk::MemoryBarrier { vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead };
            _commandBuffer.pipelineBarrier( vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits::eByRegion, memoryBarrier, nullptr, nullptr );

            _commandBuffer.bindPipeline( vk::PipelineBindPoint::eCompute, _pipelines.integralImage );
            _commandBuffer.bindDescriptorSets( vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, _descriptorSet, nullptr );
            _commandBuffer.dispatch( TextureSize, 1, 1 );

            // --- Compute integral triangles --- //
            _commandBuffer.bindPipeline( vk::PipelineBindPoint::eCompute, _pipelines.upperLeftIntegralTriangle );
            _commandBuffer.bindDescriptorSets( vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, _descriptorSet, nullptr );
            _commandBuffer.dispatch( 2 * TextureSize - 1, 1, 1 );

            _commandBuffer.bindPipeline( vk::PipelineBindPoint::eCompute, _pipelines.upperRightIntegralTriangle );
            _commandBuffer.bindDescriptorSets( vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, _descriptorSet, nullptr );
            _commandBuffer.dispatch( 2 * TextureSize - 1, 1, 1 );

            // --- Compute deformation --- //
            memoryBarrier = vk::MemoryBarrier { vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead };
            _commandBuffer.pipelineBarrier( vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits::eByRegion, memoryBarrier, nullptr, nullptr );

            _commandBuffer.bindPipeline( vk::PipelineBindPoint::eCompute, _pipelines.deformation );
            _commandBuffer.bindDescriptorSets( vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, _descriptorSet, nullptr );
            _commandBuffer.dispatch( TextureSize, 1, 1 );

            // --- Perform regularization --- //
            memoryBarrier = vk::MemoryBarrier { vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead };
            _commandBuffer.pipelineBarrier( vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits::eByRegion, memoryBarrier, nullptr, nullptr );

            const uint32_t workGroupCount { ( pointCount + 1024 - 1 ) / 1024 };
            _commandBuffer.bindPipeline( vk::PipelineBindPoint::eCompute, _pipelines.regularization );
            _commandBuffer.bindDescriptorSets( vk::PipelineBindPoint::eCompute, _pipelineLayout, 0, _descriptorSet, nullptr );
            _commandBuffer.pushConstants( _pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof( PushConstants ), &pushConstants );
            _commandBuffer.dispatch( workGroupCount, 1, 1 );

            // NOTE: If required, perform regularization of additional points here

            memoryBarrier = vk::MemoryBarrier { vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead };
            _commandBuffer.pipelineBarrier( vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eVertexShader, vk::DependencyFlagBits::eByRegion, memoryBarrier, nullptr, nullptr );
        }

        _commandBuffer.writeTimestamp( vk::PipelineStageFlagBits::eComputeShader, _queryPool, 1 );
        _commandBuffer.end();

        _queue.submit( vk::SubmitInfo{ nullptr, nullptr, _commandBuffer, nullptr } );
        _queue.waitIdle();

        const auto timestamps { _device.getQueryPoolResults<uint64_t>( _queryPool, 0u, 2u, 2 * sizeof( uint64_t ), sizeof( uint64_t ), vk::QueryResultFlagBits::e64 ).value };
        const float timestampPeriod { _physicalDevice.getProperties().limits.timestampPeriod };
        const double nanoseconds { ( timestamps[1] - timestamps[0] ) * timestampPeriod };
        return nanoseconds;
    }

private:
    void createImages()
    {
        // --- Create images --- //
        const vk::Extent3D imageExtent { TextureSize, TextureSize, 1 };
        _images.density.createColorImage( _device, _physicalDevice, imageExtent, vk::Format::eR32Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage );
        _images.integralColumns.createColorImage( _device, _physicalDevice, imageExtent, vk::Format::eR32G32Sfloat, vk::ImageUsageFlagBits::eStorage );
        _images.integralImage.createColorImage( _device, _physicalDevice, imageExtent, vk::Format::eR32Sfloat, vk::ImageUsageFlagBits::eStorage );
        _images.upperLeftIntegralTriangle.createColorImage( _device, _physicalDevice, imageExtent, vk::Format::eR32G32Sfloat, vk::ImageUsageFlagBits::eStorage );
        _images.upperRightIntegralTriangle.createColorImage( _device, _physicalDevice, imageExtent, vk::Format::eR32G32Sfloat, vk::ImageUsageFlagBits::eStorage );
        _images.deformation.createColorImage( _device, _physicalDevice, imageExtent, vk::Format::eR32G32Sfloat, vk::ImageUsageFlagBits::eStorage );

        // --- Transisiton image layouts to VK_IMAGE_LAYOUT_GENERAL --- //
        const auto imageMemoryBarrier = vk::ImageMemoryBarrier {
            vk::AccessFlagBits::eNone,
            vk::AccessFlagBits::eNone,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eGeneral,
            0,
            0,
            vk::Image{},
            vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
        };
        std::vector<vk::ImageMemoryBarrier> imageMemoryBarriers( 6, imageMemoryBarrier );
        imageMemoryBarriers[0].image = _images.density.image();
        imageMemoryBarriers[1].image = _images.integralColumns.image();
        imageMemoryBarriers[2].image = _images.integralImage.image();
        imageMemoryBarriers[3].image = _images.upperLeftIntegralTriangle.image();
        imageMemoryBarriers[4].image = _images.upperRightIntegralTriangle.image();
        imageMemoryBarriers[5].image = _images.deformation.image();

        _commandBuffer.reset();
        _commandBuffer.begin( vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit } );
        _commandBuffer.pipelineBarrier( vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eTopOfPipe, vk::DependencyFlagBits::eByRegion, nullptr, nullptr, imageMemoryBarriers );
        _commandBuffer.end();

        _queue.submit( vk::SubmitInfo{ nullptr, nullptr, _commandBuffer, nullptr } );
        _queue.waitIdle();
    }
    void createPipelines()
    {
        // --- Accumulation --- //
        const auto shaderModules = std::vector<vk::ShaderModule> {
            util::createShaderModule( _device, "./shaders/accumulation.vert.spv" ),
            util::createShaderModule( _device, "./shaders/accumulation.frag.spv" )
        };
        const auto shaderStageCreateInfos = std::vector<vk::PipelineShaderStageCreateInfo> {
            vk::PipelineShaderStageCreateInfo { vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eVertex, shaderModules[0], "main", nullptr, nullptr },
            vk::PipelineShaderStageCreateInfo { vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eFragment, shaderModules[1], "main", nullptr, nullptr }
        };
        const auto vertexInputStateCreateInfo = vk::PipelineVertexInputStateCreateInfo { vk::PipelineVertexInputStateCreateFlags{}, nullptr, nullptr, nullptr };
        const auto inputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo { vk::PipelineInputAssemblyStateCreateFlags{}, vk::PrimitiveTopology::ePointList, false, nullptr };
        const auto viewport = vk::Viewport { 0.0f, 0.0f, static_cast<float>( TextureSize ), static_cast<float>( TextureSize ), 0.0f, 1.0f };
        const auto scissor = vk::Rect2D { vk::Offset2D { 0, 0 }, vk::Extent2D { TextureSize, TextureSize } };
        const auto viewportStateCreateInfo = vk::PipelineViewportStateCreateInfo { vk::PipelineViewportStateCreateFlags{}, viewport, scissor, nullptr };
        const auto rasterizationStateCreateInfo = vk::PipelineRasterizationStateCreateInfo {
            vk::PipelineRasterizationStateCreateFlags{},
            false,
            false,
            vk::PolygonMode::eFill,
            vk::CullModeFlagBits::eNone,
            vk::FrontFace::eClockwise,
            false,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            nullptr
        };
        const auto colorBlendAttachmentStates = std::vector<vk::PipelineColorBlendAttachmentState> {
            vk::PipelineColorBlendAttachmentState { true, vk::BlendFactor::eOne, vk::BlendFactor::eOne, vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR }
        };
        const auto colorBlendStateCreateInfo = vk::PipelineColorBlendStateCreateInfo {
            vk::PipelineColorBlendStateCreateFlags{},
            false,
            vk::LogicOp::eNoOp,
            colorBlendAttachmentStates,
            std::array<float, 4>{},
            nullptr
        };
        const auto accumulationPipelineCreateInfo = vk::GraphicsPipelineCreateInfo {
            vk::PipelineCreateFlags{},
            shaderStageCreateInfos,
            &vertexInputStateCreateInfo,
            &inputAssemblyStateCreateInfo,
            nullptr,
            &viewportStateCreateInfo,
            &rasterizationStateCreateInfo,
            nullptr,
            nullptr,
            &colorBlendStateCreateInfo,
            nullptr,
            _pipelineLayout,
            _renderPass,
            0,
            vk::Pipeline{},
            0,
            nullptr
        };
        _pipelines.accumulation = _device.createGraphicsPipeline( _pipelineCache, accumulationPipelineCreateInfo ).value;
        for( const auto shaderModule : shaderModules )
        {
            _device.destroyShaderModule( shaderModule );
        }

        // --- Gaussian kernel x --- //
        auto computePipelineCreateInfo = vk::ComputePipelineCreateInfo {
            vk::PipelineCreateFlags{},
            vk::PipelineShaderStageCreateInfo{ vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, util::createShaderModule( _device, "./shaders/gaussian_kernel_x.comp.spv" ), "main", nullptr, nullptr },
            _pipelineLayout,
        };
        _pipelines.gaussianKernelX = _device.createComputePipeline( _pipelineCache, computePipelineCreateInfo ).value;
        _device.destroyShaderModule( computePipelineCreateInfo.stage.module );

        // --- Gaussian kernel y --- //
        computePipelineCreateInfo = vk::ComputePipelineCreateInfo {
            vk::PipelineCreateFlags{},
            vk::PipelineShaderStageCreateInfo{ vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, util::createShaderModule( _device, "./shaders/gaussian_kernel_y.comp.spv" ), "main", nullptr, nullptr },
            _pipelineLayout,
        };
        _pipelines.gaussianKernelY = _device.createComputePipeline( _pipelineCache, computePipelineCreateInfo ).value;
        _device.destroyShaderModule( computePipelineCreateInfo.stage.module );

        // --- Integral columns --- //
        computePipelineCreateInfo = vk::ComputePipelineCreateInfo {
            vk::PipelineCreateFlags{},
            vk::PipelineShaderStageCreateInfo{ vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, util::createShaderModule( _device, "./shaders/integral_columns.comp.spv" ), "main", nullptr, nullptr },
            _pipelineLayout,
        };
        _pipelines.integralColumns = _device.createComputePipeline( _pipelineCache, computePipelineCreateInfo ).value;
        _device.destroyShaderModule( computePipelineCreateInfo.stage.module );

        // --- Integral image --- //
        computePipelineCreateInfo = vk::ComputePipelineCreateInfo {
            vk::PipelineCreateFlags{},
            vk::PipelineShaderStageCreateInfo{ vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, util::createShaderModule( _device, "./shaders/integral_image.comp.spv" ), "main", nullptr, nullptr },
            _pipelineLayout,
        };
        _pipelines.integralImage = _device.createComputePipeline( _pipelineCache, computePipelineCreateInfo ).value;
        _device.destroyShaderModule( computePipelineCreateInfo.stage.module );

        // --- Upper-left integral triangle --- //
        computePipelineCreateInfo = vk::ComputePipelineCreateInfo {
            vk::PipelineCreateFlags{},
            vk::PipelineShaderStageCreateInfo{ vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, util::createShaderModule( _device, "./shaders/upper_left_integral_triangle.comp.spv" ), "main", nullptr, nullptr },
            _pipelineLayout,
        };
        _pipelines.upperLeftIntegralTriangle = _device.createComputePipeline( _pipelineCache, computePipelineCreateInfo ).value;
        _device.destroyShaderModule( computePipelineCreateInfo.stage.module );

        // --- Upper-right integral triangle --- //
        computePipelineCreateInfo = vk::ComputePipelineCreateInfo {
            vk::PipelineCreateFlags{},
            vk::PipelineShaderStageCreateInfo{ vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, util::createShaderModule( _device, "./shaders/upper_right_integral_triangle.comp.spv" ), "main", nullptr, nullptr },
            _pipelineLayout,
        };
        _pipelines.upperRightIntegralTriangle = _device.createComputePipeline( _pipelineCache, computePipelineCreateInfo ).value;
        _device.destroyShaderModule( computePipelineCreateInfo.stage.module );

        // --- Deformation image --- //
        computePipelineCreateInfo = vk::ComputePipelineCreateInfo {
            vk::PipelineCreateFlags{},
            vk::PipelineShaderStageCreateInfo{ vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, util::createShaderModule( _device, "./shaders/deformation.comp.spv" ), "main", nullptr, nullptr },
            _pipelineLayout,
        };
        _pipelines.deformation = _device.createComputePipeline( _pipelineCache, computePipelineCreateInfo ).value;
        _device.destroyShaderModule( computePipelineCreateInfo.stage.module );

        // --- Regularization --- //
        computePipelineCreateInfo = vk::ComputePipelineCreateInfo {
            vk::PipelineCreateFlags{},
            vk::PipelineShaderStageCreateInfo{ vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, util::createShaderModule( _device, "./shaders/regularization.comp.spv" ), "main", nullptr, nullptr },
            _pipelineLayout,
        };
        _pipelines.regularization = _device.createComputePipeline( _pipelineCache, computePipelineCreateInfo ).value;
        _device.destroyShaderModule( computePipelineCreateInfo.stage.module );
    }

    vk::Instance _instance {};

    vk::PhysicalDevice _physicalDevice {};
    uint32_t _queueFamilyIndex {};
    uint32_t _timestampValidBits {};

    vk::Device _device {};
    vk::Queue _queue {};

    vk::QueryPool _queryPool {};

    vk::CommandPool _commandPool {};
    vk::CommandBuffer _commandBuffer {};

    struct
    {
        util::ImageWrapper density {};
        util::ImageWrapper integralColumns {};
        util::ImageWrapper integralImage {};
        util::ImageWrapper upperLeftIntegralTriangle {};
        util::ImageWrapper upperRightIntegralTriangle {};
        util::ImageWrapper deformation {};
    } _images {};

    vk::RenderPass _renderPass {};
    vk::Framebuffer _framebuffer {};

    vk::Sampler _sampler {};

    vk::DescriptorSetLayout _descriptorSetLayout {};
    vk::DescriptorPool _descriptorPool {};
    vk::DescriptorSet _descriptorSet {};

    vk::PipelineLayout _pipelineLayout {};
    struct PushConstants
    {
        uint32_t pointCount {};
        uint32_t kernelRadius {};
    };

    vk::PipelineCache _pipelineCache {};
    struct
    {
        vk::Pipeline accumulation {};
        vk::Pipeline gaussianKernelX {};
        vk::Pipeline gaussianKernelY {};
        vk::Pipeline integralColumns {};
        vk::Pipeline integralImage {};
        vk::Pipeline upperLeftIntegralTriangle {};
        vk::Pipeline upperRightIntegralTriangle {};
        vk::Pipeline deformation {};
        vk::Pipeline regularization {};
    } _pipelines {};
};